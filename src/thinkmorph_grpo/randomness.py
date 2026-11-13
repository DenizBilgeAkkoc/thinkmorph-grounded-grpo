"""Seed initialization separately from per-rollout random streams."""
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    """A weights_only-safe snapshot, including NumPy's generator state."""
    name, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return {"python": random.getstate(), "numpy": {
        "name": name, "keys": torch.from_numpy(keys.astype(np.int64)),
        "position": position, "has_gauss": has_gauss, "cached_gauss": cached_gauss},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns["name"], ns["keys"].cpu().numpy().astype(np.uint32),
                         ns["position"], ns["has_gauss"], ns["cached_gauss"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG topology differs from this process")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
