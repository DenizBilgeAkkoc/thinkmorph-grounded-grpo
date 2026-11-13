"""Standard SDE and summed Gaussian likelihood from custom BAGEL-Canvas GRPO.

Source: modeling/bagel/bagel.py::_sde_step_with_logprob, standard 'sde' branch.
Deterministic transitions have no likelihood and are excluded from PPO.
"""
import torch


def schedule(points=15, shift=3.0, device="cpu"):
    if points < 4 or shift <= 0:
        raise ValueError("Invalid schedule")
    t = torch.linspace(1, 0, points, device=device)
    return shift * t / (1 + (shift - 1) * t)


def ode_step(velocity, t, next_t, sample):
    """Custom SDE kernel's noise=0 branch, without its undefined Gaussian score."""
    return sample.float() + velocity.float() * (next_t - t)


def sde_step(velocity, t, next_t, sample, *, sigma_max, noise_level=1.0,
             previous_sample=None, generator=None):
    if noise_level <= 0:
        raise ValueError("Use an ODE transition outside the stochastic window")
    if previous_sample is not None and generator is not None:
        raise ValueError("Cannot pass both generator and previous_sample")
    velocity, sample = velocity.float(), sample.float()
    t = torch.as_tensor(t, device=sample.device, dtype=torch.float32)
    next_t = torch.as_tensor(next_t, device=sample.device, dtype=torch.float32)
    sigma_max = torch.as_tensor(sigma_max, device=sample.device, dtype=torch.float32)
    dt = next_t - t
    sigma = torch.sqrt(t / (1 - torch.where(t == 1, sigma_max, t))) * noise_level
    mean = sample * (1 + sigma.square() / (2 * t) * dt)
    mean = mean + velocity * (1 + sigma.square() * (1 - t) / (2 * t)) * dt
    std = sigma * torch.sqrt(-dt)
    if previous_sample is None:
        noise = torch.randn(velocity.shape, device=velocity.device,
                            dtype=velocity.dtype, generator=generator)
        previous_sample = mean + std * noise
    logp = -(previous_sample.float().detach() - mean).square() / (2 * std.square())
    return previous_sample, logp.sum(), mean, sigma


def shared_windows(group_seed, max_rounds=3, points=15, width=3):
    """Same window for every member of a group at each generated-image position."""
    high = points // 2 - width + 1
    if width < 1 or high < 1:
        raise ValueError("SDE window does not fit the custom trainer's first-half range")
    g = torch.Generator(device="cpu").manual_seed(group_seed)
    return torch.randint(0, high, (max_rounds,), generator=g).tolist()
