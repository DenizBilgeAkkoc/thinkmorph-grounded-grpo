"""Custom GRPO reductions, with explicit masks for native variable-length traces."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class GRPOConfig:
    gamma: float = .7
    text_weight: float = .3
    accuracy_weight: float = .5
    min_std: float = .02
    adv_clip: float = 5.0
    clip_range: float = .2
    velocity_beta: float = 1e-4
    normalize_process_returns: bool = False
    unfinished_penalty: float = 0.0


def normalize_advantages(rewards, min_std, eps=1e-8):
    if rewards.numel() < 2 or rewards.std().item() < min_std:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def group_returns(image_rewards, text_rewards, valid, terminal, cfg, *, image_valid=None, completed=None):
    """[N,S] rewards; native early termination creates right padding, never fake steps."""
    if image_rewards.shape != text_rewards.shape or valid.shape != image_rewards.shape:
        raise ValueError("Reward/mask shapes must agree")
    if valid.dtype != torch.bool or terminal.shape != (valid.shape[0],):
        raise ValueError("Invalid trajectory mask or terminal shape")
    if torch.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("Native traces must be a valid prefix followed by padding")
    image_valid = valid if image_valid is None else image_valid
    if image_valid.shape != valid.shape or image_valid.dtype != torch.bool or torch.any(image_valid & ~valid):
        raise ValueError("Invalid image-action mask")
    if not 0 <= cfg.gamma <= 1 or cfg.unfinished_penalty < 0:
        raise ValueError("Invalid return configuration")
    im, tx = image_rewards * image_valid, text_rewards * valid
    im_weight, tx_weight = image_valid.float(), valid.float()
    for s in range(valid.shape[1] - 2, -1, -1):
        im[:, s] += cfg.gamma * im[:, s + 1] * valid[:, s + 1]
        tx[:, s] += cfg.gamma * tx[:, s + 1] * valid[:, s + 1]
        im_weight[:, s] += cfg.gamma * im_weight[:, s + 1] * valid[:, s + 1]
        tx_weight[:, s] += cfg.gamma * tx_weight[:, s + 1] * valid[:, s + 1]
    if cfg.normalize_process_returns:
        im = im / im_weight.clamp_min(1e-8)
        tx = tx / tx_weight.clamp_min(1e-8)
    combined = im + cfg.text_weight * tx + cfg.accuracy_weight * terminal[:, None]
    if cfg.unfinished_penalty:
        if completed is None or completed.shape != terminal.shape or completed.dtype != torch.bool:
            raise ValueError("Completion mask required for unfinished-trajectory penalty")
        combined -= cfg.unfinished_penalty * (~completed)[:, None]
    return combined * valid


def group_advantages(image_rewards, text_rewards, valid, terminal, cfg, *, image_valid=None, completed=None):
    combined = group_returns(image_rewards, text_rewards, valid, terminal, cfg,
        image_valid=image_valid, completed=completed)
    advantages = torch.zeros_like(combined)
    for s in range(valid.shape[1]):
        mask = valid[:, s]
        advantages[mask, s] = normalize_advantages(combined[mask, s], cfg.min_std)
    return advantages.clamp(-cfg.adv_clip, cfg.adv_clip)


def ppo_loss(new_logp, old_logp, advantage, cfg):
    ratio = torch.exp((new_logp - old_logp.detach()).clamp(-20, 20))
    adv = torch.as_tensor(advantage, device=ratio.device).clamp(-cfg.adv_clip, cfg.adv_clip)
    loss = torch.maximum(-adv * ratio, -adv * ratio.clamp(1-cfg.clip_range, 1+cfg.clip_range))
    return loss.mean(), ratio.detach()


def velocity_regularizer(current, reference):
    # Custom trainer calls this KL, but it is summed velocity MSE, not categorical KL.
    return (current.float() - reference.detach().float()).square().sum()
