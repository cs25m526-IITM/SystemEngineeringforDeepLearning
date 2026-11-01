"""CIFAR-10 data pipeline: normalisation, augmentation, train/val/test loaders."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .utils import seed_worker

# Per-channel statistics computed on the CIFAR-10 training set.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms():
    """Return (train_transform, eval_transform).

    Train: random crop (pad 4) + horizontal flip + ToTensor + Normalize.
    Eval : ToTensor + Normalize only (no augmentation).
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, eval_tf


def get_loaders(data_dir: str, batch_size: int, seed: int, val_size: int = 5000,
                num_workers: int = 2):
    """45k train / 5k validation (fixed split by seed) / 10k test."""
    train_tf, eval_tf = get_transforms()

    # Two views of the same training set: augmented for training, clean for validation.
    train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    val_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)

    loader_kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                     worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **loader_kw)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, **loader_kw)
    return train_loader, val_loader, test_loader
