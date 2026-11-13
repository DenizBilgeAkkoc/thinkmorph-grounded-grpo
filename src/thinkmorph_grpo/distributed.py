"""Synchronous, bucketed gradient averaging for the native replay trainer.

Native replay invokes many different forwards/backwards and frozen references.
Synchronize only once after complete local validation; do not wrap those native
calls in DDP or communicate after every small text/flow action.
"""
import hashlib
import math
import time
import torch
import torch.distributed as dist


def synchronize_gradients(model, denominator, bucket_bytes=16 * 1024 * 1024, process_group=None):
    """Sum local question gradients and divide by assigned, scored questions.

    Flat questions contribute zeros and count in the denominator. Unassigned
    or explicitly skipped questions contribute zeros and do not count. A
    globally unused parameter keeps grad=None, preserving AdamW semantics.
    All peers must call in the same parameter order and with the same divisor.
    """
    started = time.monotonic()
    parameters = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters to synchronize")
    active = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size(process_group) if active else 1
    device = parameters[0][1].device

    def reduce(tensor, op=dist.ReduceOp.SUM):
        if active:
            dist.all_reduce(tensor, op=op, group=process_group)

    valid_divisor = isinstance(denominator,(int,float)) and math.isfinite(denominator) and denominator > 0
    local_bad = (not valid_divisor or not isinstance(bucket_bytes,int) or bucket_bytes <= 0
                 or any(p.device != device or (p.grad is not None and
                     (p.grad.is_sparse or p.grad.shape != p.shape or p.grad.dtype != p.dtype))
                        for _,p in parameters))
    bad = torch.tensor(int(local_bad), dtype=torch.int32, device=device)
    reduce(bad, dist.ReduceOp.MAX)
    if bad.item():
        raise ValueError("Invalid distributed gradient, denominator, bucket size or device")
    schema = hashlib.sha256(repr([(name,tuple(p.shape),str(p.dtype)) for name,p in parameters]).encode()).digest()
    divisor_min = torch.tensor([float(denominator),bucket_bytes,*schema],dtype=torch.float64,device=device)
    divisor_max = divisor_min.clone()
    reduce(divisor_min,dist.ReduceOp.MIN)
    reduce(divisor_max,dist.ReduceOp.MAX)
    if not torch.equal(divisor_min,divisor_max):
        raise ValueError("Distributed gradient denominators, bucket sizes or parameter schemas differ")
    # Check all grads together, reducing one boolean per rank rather than
    # synchronizing the GPU for every small LoRA matrix.
    finite = torch.ones((),dtype=torch.bool,device=device)
    for _,p in parameters:
        if p.grad is not None:
            finite.logical_and_(torch.isfinite(p.grad).all())
    bad = (~finite).to(torch.int32)
    reduce(bad,dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError("Nonfinite gradient on a distributed rank; no update committed")
    used = torch.tensor([p.grad is not None for _,p in parameters],dtype=torch.int32,device=device)
    reduce(used,dist.ReduceOp.MAX)
    used_list = used.cpu().tolist()
    buckets, current, size, dtype = [], [], 0, None
    for use,(_,p) in zip(used_list,parameters):
        if not use:
            p.grad = None
            continue
        nbytes = p.numel()*p.element_size()
        if current and (p.dtype != dtype or size+nbytes > bucket_bytes):
            buckets.append(current)
            current,size = [],0
        current.append(p)
        size += nbytes
        dtype = p.dtype
    if current:
        buckets.append(current)
    communicated = 0
    for bucket in buckets:
        buffer = torch.cat([p.grad.detach().reshape(-1) if p.grad is not None else
                            torch.zeros(p.numel(),dtype=p.dtype,device=p.device) for p in bucket])
        reduce(buffer)
        buffer.div_(denominator)
        offset=0
        for p in bucket:
            averaged=buffer[offset:offset+p.numel()].view_as(p)
            if p.grad is None:
                p.grad=averaged.clone()
            else:
                p.grad.copy_(averaged)
            offset += p.numel()
        communicated += buffer.numel()*buffer.element_size()
        finite.logical_and_(torch.isfinite(buffer).all())
    bad=(~finite).to(torch.int32)
    reduce(bad,dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError("Nonfinite averaged gradients; no update committed")
    return dict(world_size=world,question_denominator=denominator,buckets=len(buckets),
                gradient_payload_bytes=communicated,globally_used_parameters=sum(used_list),
                globally_unused_parameters=len(used_list)-sum(used_list),
                synchronization_seconds=time.monotonic()-started)
