"""Isolated RTX execution experiments; no reward or optimizer-rule changes."""
import os
from types import MethodType


def execution_settings():
    return dict(generation_text_batch_size=int(os.environ.get('THINKMORPH_TEXT_BATCH_SIZE','2')),
                replay_text_batch_size=2,image_batch_size=int(os.environ.get('THINKMORPH_IMAGE_BATCH_SIZE','1')),layer_checkpointing=os.environ.get('THINKMORPH_REPLAY_CHECKPOINT','on'),
                vae_device=os.environ.get('THINKMORPH_VAE_DEVICE','cpu'),vae_precision='float32',
                vae_posterior_rng_device='cpu',checkpoint_token_threshold=2048)


def replay_batches(batches, width=2):
    """Keep original logical pairs for scoring, independent of generation width.

    Each text action remains exactly once in the N8 objective. Group by fixed
    logical rollout pair, then text action, matching the production pair order.
    Old and new probabilities use the same resulting layout and full scorer.
    """
    if width!=2: raise ValueError('This trial retains the production replay width of 2')
    groups={};seen=set()
    for batch in batches:
        for member in batch['members']:
            identity=(member['rollout_index'],member['text_action_index'])
            if identity in seen: raise ValueError('Duplicate original text action')
            seen.add(identity)
            groups.setdefault((identity[0]//width,identity[1]),[]).append((member,batch))
    result=[]
    for (pair,action),entries in sorted(groups.items()):
        entries.sort(key=lambda item:item[0]['rollout_index'])
        if not 1<=len(entries)<=width: raise ValueError('Invalid replay pair membership')
        exemplar=dict(entries[0][1]);exemplar['members']=[m for m,_ in entries]
        counts=[m['event']['sampled_tokens'] for m,_ in entries]
        exemplar['active_lane_schedule']=[[i for i,c in enumerate(counts) if step<c] for step in range(max(counts))]
        exemplar['sampling_batch_ids']=sorted({b['batch_id'] for _,b in entries})
        exemplar['batch_id']=f'replay_pair{pair}_action{action}'
        result.append(exemplar)
    actual=[(m['rollout_index'],m['text_action_index']) for b in result for m in b['members']]
    if len(actual)!=len(set(actual)) or set(actual)!=seen: raise ValueError('Replay lost original text actions')
    return result


def configure_vae(vae):
    """Move only VAE convolutions to GPU, keeping posterior RNG on CPU.

    FP32, TF32 disabled. Preserve native CPU posterior sampling, scale/shift,
    output dtype/device, and CPU RNG consumption. Encoder/decoder hardware can
    introduce small floating-point differences; those are measured separately.
    Actual conditioning latents remain recorded and reused by native replay.
    """
    device=os.environ.get('THINKMORPH_VAE_DEVICE','cpu')
    if device=='cpu':return
    if device!='cuda':raise ValueError('VAE device must be cpu or cuda')
    import torch
    if getattr(vae,'_rtx_cuda_vae',False):return
    vae.encoder.to(device='cuda',dtype=torch.float32)
    vae.decoder.to(device='cuda',dtype=torch.float32)
    def encode(self,x):
        if x.device.type!='cpu' or x.dtype!=torch.float32:raise ValueError('Expected native FP32 CPU VAE input')
        with torch.no_grad(),torch.autocast('cuda',enabled=False),torch.backends.cudnn.flags(allow_tf32=False):
            stats=self.encoder(x.to('cuda')).cpu()
        z=self.reg(stats)
        return self.scale_factor*(z-self.shift_factor)
    def decode(self,z):
        if z.device.type!='cpu':raise ValueError('Expected native CPU latent')
        z=z/self.scale_factor+self.shift_factor
        with torch.no_grad(),torch.autocast('cuda',enabled=False),torch.backends.cudnn.flags(allow_tf32=False):
            return self.decoder(z.to(device='cuda',dtype=torch.float32)).cpu()
    vae.encode=MethodType(encode,vae);vae.decode=MethodType(decode,vae)
    vae._rtx_cuda_vae=True
