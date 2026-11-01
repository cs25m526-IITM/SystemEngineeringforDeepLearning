"""Small helpers: seeding, device selection, run naming."""
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG we rely on so a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels (slightly slower, but reproducible).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers (used together with a seeded Generator)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_name(cfg) -> str:
    return (f"act_{cfg.activation}_bs_{cfg.batch_size}_opt_{cfg.optimizer}"
            f"_lr_{cfg.lr}_wd_{cfg.weight_decay}_bn_{cfg.batch_norm}_ep_{cfg.epochs}")
