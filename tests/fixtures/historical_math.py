"""Independent frozen test oracle from the original BAGEL-Canvas trainer.
Used through AST extraction; this module is not imported by the training code.
"""

# Frozen historical oracle: BAGEL-Canvas/scripts/train/train_grpo_v2_batched.py::make_grpo_config
def make_grpo_config(args):
    cfg = SimpleNamespace()
    cfg.train = SimpleNamespace(adv_clip_max=args.adv_clip_max, clip_range_lt=args.clip_range, clip_range_gt=args.clip_range_high if args.clip_range_high is not None else args.clip_range, beta=args.kl_beta)
    cfg.sample = SimpleNamespace(sde_window_size=args.sde_window_size, sde_window_range=(0, args.num_timesteps // 2))
    if args.task == 'zebra_multistep' and hasattr(args, 'zebra_cfg_text_scale'):
        cfg.sample.guidance = dict(cfg_text_scale=args.zebra_cfg_text_scale, cfg_img_scale=args.zebra_cfg_img_scale, cfg_interval=(0.4, 1.0), cfg_renorm_type='channel')
    return cfg

# Frozen historical oracle: BAGEL-Canvas/train/grpo_inferencer.py::sample_guidance
def sample_guidance(self, grpo_config, text_context, image_context, image_shape):
    settings = getattr(getattr(grpo_config, 'sample', None), 'guidance', {})
    if not settings:
        return {}
    return dict(settings, **self.guidance_inputs(text_context, image_context, image_shape))

# Frozen historical oracle: BAGEL-Canvas/modeling/bagel/bagel.py::_sde_step_with_logprob
def _sde_step_with_logprob(self, model_output: torch.Tensor, timestep: torch.Tensor, prev_timestep: torch.Tensor, d_timestep: torch.Tensor, sample: torch.Tensor, prev_sample: Optional[torch.Tensor]=None, generator: Optional[torch.Generator]=None, sigma_max: Optional[float]=None, noise_level: float=0.8, sde_type: str='sde'):
    """
        SDE step with log probability computation for GRPO training.
        Converts ODE flow to SDE by injecting noise and computes log probabilities.

        Args:
            model_output: Velocity prediction from the model
            timestep: Current timestep
            prev_timestep: Previous timestep
            d_timestep: Delta timestep (prev_timestep - timestep)
            sample: Current latent sample
            prev_sample: Optional previous sample (for training)
            generator: Random generator for reproducibility
            sigma_max: Maximum sigma value
            noise_level: Controls amount of noise injection
            sde_type: "sde" for standard SDE, "cps" for Coefficients-Preserving Sampling
        """
    import math
    from diffusers.utils.torch_utils import randn_tensor
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()
    if sde_type == 'sde':
        std_dev_t = torch.sqrt(timestep / (1 - torch.where(timestep == 1, sigma_max, timestep))) * noise_level
        prev_sample_mean = sample * (1 + std_dev_t ** 2 / (2 * timestep) * d_timestep) + model_output * (1 + std_dev_t ** 2 * (1 - timestep) / (2 * timestep)) * d_timestep
        if prev_sample is not None and generator is not None:
            raise ValueError('Cannot pass both generator and prev_sample.')
        if prev_sample is None:
            variance_noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * d_timestep) * variance_noise
        log_prob = -(prev_sample.detach() - prev_sample_mean) ** 2 / (2 * (std_dev_t * torch.sqrt(-1 * d_timestep)) ** 2)
    elif sde_type == 'cps':
        std_dev_t = prev_timestep * math.sin(noise_level * math.pi / 2)
        pred_original_sample = sample - timestep * model_output
        noise_estimate = sample + model_output * (1 - timestep)
        prev_sample_mean = pred_original_sample * (1 - prev_timestep) + noise_estimate * torch.sqrt(prev_timestep ** 2 - std_dev_t ** 2)
        if prev_sample is None:
            variance_noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
            prev_sample = prev_sample_mean + std_dev_t * variance_noise
        log_prob = -(prev_sample.detach() - prev_sample_mean) ** 2
    else:
        raise ValueError(f'Unknown sde_type: {sde_type}')
    log_prob = log_prob.sum()
    return (prev_sample, log_prob, prev_sample_mean, std_dev_t)

# Frozen historical oracle: BAGEL-Canvas/train/grpo_utils.py::normalize_advantages
def normalize_advantages(rewards, min_std, eps=1e-08):
    """Normalise advantages within this rank's rollouts for a single prompt."""
    if rewards.numel() < 2 or rewards.std().item() < min_std:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (rewards.std() + eps)
