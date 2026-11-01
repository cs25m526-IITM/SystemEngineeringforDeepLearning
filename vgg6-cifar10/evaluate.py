"""Evaluate a saved checkpoint on the CIFAR-10 test set (and validation split)."""
import argparse

import torch
import torch.nn as nn

from src.data import get_loaders
from src.engine import evaluate
from src.models import VGG6
from src.utils import get_device, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--data_dir", default="./data")
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    set_seed(cfg["seed"])

    _, val_loader, test_loader = get_loaders(args.data_dir, cfg["batch_size"], cfg["seed"])
    model = VGG6(cfg["activation"], cfg["batch_norm"]).to(device)
    model.load_state_dict(ckpt["model"])
    crit = nn.CrossEntropyLoss()
    _, va = evaluate(model, val_loader, crit, device)
    _, te = evaluate(model, test_loader, crit, device)
    print(f"Config: {cfg}")
    print(f"Validation top-1: {va:.2f}% | Test top-1: {te:.2f}%")


if __name__ == "__main__":
    main()
