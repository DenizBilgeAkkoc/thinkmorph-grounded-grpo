"""Accumulate complete question objectives without pooling their N8 rewards.

All questions must have been sampled at the same current policy. This creates
one optimizer batch containing multiple questions, not consecutive policy
updates. Distributed callers request a sum, then divide once by the GLOBAL
accepted-question count using the existing gradient synchronizer.
"""
from pathlib import Path
import torch
from .parallel_trainer import _parameter_versions

def accumulate_questions(trainer,questions,*,normalize=True):
    if not 1<=len(questions)<=4:raise ValueError('Expected one to four complete questions')
    seen=set()
    for q in questions:
        folders=[str(Path(p).resolve()) for p in q['folders']]
        if len(folders)!=8 or len(q['judgments'])!=8 or len(set(folders))!=8 or seen.intersection(folders):
            raise ValueError('Questions must contain disjoint complete N8 groups')
        seen.update(folders)
    model=trainer.model;parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    versions=_parameter_versions(model);accumulated={};metrics=[]
    trainer.optimizer.zero_grad(set_to_none=True)
    try:
        for question in questions:
            result=trainer.update(question['folders'],question['judgments'],optimize=True,defer_step=True)
            if result.get('optimized') or _parameter_versions(model)!=versions:
                raise ValueError('Policy changed before all questions finished replay')
            metrics.append(result)
            for name,p in parameters.items():
                if p.grad is None:continue
                if not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite question gradient')
                if name not in accumulated:accumulated[name]=p.grad.detach().float().clone()
                else:accumulated[name].add_(p.grad.detach().float())
            trainer.optimizer.zero_grad(set_to_none=True)
        for name,value in accumulated.items():
            if normalize:value.div_(len(questions))
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite accumulated gradient')
            parameters[name].grad=value.to(dtype=parameters[name].dtype)
        return dict(questions=len(questions),reward_group_size=8,question_metrics=metrics,
            gradients_ready=bool(accumulated),optimized=False,
            normalization='mean_over_accepted_questions' if normalize else 'sum_for_global_question_count_division',
            flat_questions=sum(not m.get('gradients_ready',False) for m in metrics))
    except BaseException:
        trainer.optimizer.zero_grad(set_to_none=True)
        raise
