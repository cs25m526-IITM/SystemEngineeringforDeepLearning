"""Train VGG6 on CIFAR-10 with a configurable setup; logs to Weights & Biases.

Single run : python train.py --activation gelu --optimizer nadam --lr 0.001 ...
Sweep agent: wandb agent <entity>/<project>/<sweep_id>   (see configs/sweep.yaml)
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import wandb

from src.data import get_loaders
from src.engine import evaluate, train_one_epoch
from src.models import VGG6
from src.optimizers import build_optimizer
from src.utils import get_device, make_run_name, set_seed


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "y")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--activation", default="relu",
                   choices=["relu", "sigmoid", "tanh", "silu", "gelu", "leaky_relu", "selu"])
    p.add_argument("--optimizer", default="nadam",
                   choices=["sgd", "nesterov", "adam", "adagrad", "rmsprop", "nadam"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_norm", type=str2bool, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--save_dir", default="./checkpoints")
    p.add_argument("--project", default="vgg6-cifar10")
    p.add_argument("--no_wandb", action="store_true", help="disable W&B logging")
    p.add_argument("--save_name", default=None, help="checkpoint filename (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()
    run = None
    if not args.no_wandb:
        run = wandb.init(project=args.project, config=vars(args))
        cfg = wandb.config   # sweeps override the CLI defaults here
        for k in vars(args):
            if k in cfg:
                setattr(args, k, cfg[k])
        run.name = make_run_name(args)

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | Config: {vars(args)}")

    train_loader, val_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.seed)
    model = VGG6(args.activation, args.batch_norm).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr,
                                args.weight_decay, args.momentum)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_name = args.save_name or f"{make_run_name(args)}.pt"
    ckpt_path = os.path.join(args.save_dir, ckpt_name)
    best_val, best_epoch = 0.0, -1

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"[{epoch+1:03d}/{args.epochs}] train {tr_loss:.4f}/{tr_acc:.2f}% | "
              f"val {va_loss:.4f}/{va_acc:.2f}%")
        if run:
            wandb.log({"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc,
                       "val_loss": va_loss, "val_acc": va_acc}, step=epoch)
        if va_acc > best_val:
            best_val, best_epoch = va_acc, epoch + 1
            torch.save({"model": model.state_dict(), "config": vars(args),
                        "val_acc": va_acc, "epoch": epoch + 1}, ckpt_path)

    # Final test accuracy using the best-validation checkpoint.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"Best val acc {best_val:.2f}% (epoch {best_epoch}) | Test top-1 {te_acc:.2f}%")

    if run:
        # val_acc summary = best validation accuracy (used by the parallel-coordinates plot)
        wandb.summary["val_acc"] = best_val
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["test_acc"] = te_acc
        wandb.summary["test_loss"] = te_loss
        run.finish()

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"{os.path.splitext(ckpt_name)[0]}.json"), "w") as f:
        json.dump({"config": vars(args), "best_val_acc": best_val,
                   "best_epoch": best_epoch, "test_acc": te_acc}, f, indent=2)


if __name__ == "__main__":
    main()
