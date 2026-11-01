# VGG6 on CIFAR-10 — CS6886W Assignment 1

Explores how activation, optimizer, batch size, epochs and learning rate affect a VGG6
(4 conv + 2 FC) trained on CIFAR-10. All experiments are tracked with Weights & Biases.

**Report PDF GitHub link:** `<paste your repo URL here>`

## Repository layout
```
train.py            # entry point: trains one config (also used by W&B sweep agents)
evaluate.py         # loads a checkpoint and reports val/test top-1
src/data.py         # transforms, 45k/5k/10k train/val/test loaders
src/models.py       # VGG6 with configurable activation + BatchNorm
src/optimizers.py   # SGD, Nesterov-SGD, Adam, Adagrad, RMSprop, Nadam
src/engine.py       # train_one_epoch / evaluate loops
src/utils.py        # seeding, device, run naming
configs/            # W&B sweep definitions (main, activation, optimizer, hparams)
checkpoints/        # saved models (best_model.pt = final submission model)
results/            # JSON summary per run
```

## Environment
- Python 3.10+, tested versions pinned in `requirements.txt`
  (torch 2.4.1, torchvision 0.19.1, wandb 0.18.5, numpy 1.26.4)
- GPU recommended (CUDA); CPU works but is slow.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login
```

## Seed configuration
Default `--seed 42`. It seeds Python, NumPy, PyTorch (CPU+CUDA), the DataLoader generator and
workers, the train/val split, and sets `cudnn.deterministic=True`, `cudnn.benchmark=False`.

## Data
- Split: 45,000 train / 5,000 validation (fixed by seed) / 10,000 test (official test set).
- Normalisation: mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616).
- Train augmentation: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip(0.5)`.
- Eval: only `ToTensor` + `Normalize`.

## Reproduce
**Single run (baseline):**
```bash
python train.py --activation relu --optimizer nadam --lr 0.001 --batch_size 128 \
    --epochs 30 --weight_decay 0.0005 --batch_norm True --seed 42 --save_name baseline.pt
```

**Final best configuration (Q4):** fill in after the sweep, e.g.
```bash
python train.py --activation <A> --optimizer <O> --lr <LR> --batch_size <BS> --epochs <E> \
    --momentum <M> --weight_decay <WD> --batch_norm <BN> --seed 42 --save_name best_model.pt
python evaluate.py --checkpoint checkpoints/best_model.pt
```

**Sweeps (W&B):**
```bash
wandb sweep configs/sweep.yaml              # main sweep -> parallel-coordinates plot
wandb agent <entity>/vgg6-cifar10/<sweep_id> --count 60

wandb sweep configs/sweep_activation.yaml   # Q2(a)
wandb sweep configs/sweep_optimizer.yaml    # Q2(b)
wandb sweep configs/sweep_hparams.yaml      # Q2(c)
```

## Generating the required plots (W&B UI)
- Parallel coordinates: Workspace → Add panel → *Parallel coordinates*; columns:
  activation, batch_norm, batch_size, epochs, lr, momentum, optimizer, weight_decay, val_acc.
- Val acc vs step (scatter): Add panel → *Scatter plot*, x = `Created`, y = `val_acc` (summary).
- Curves: line plots for `train_loss`, `train_acc`, `val_loss`, `val_acc`.

## Results & analysis
See the report PDF / `REPORT_TEMPLATE.md`. Fill in the table below after your runs.

| Study | Best setting | Val acc | Test acc |
|---|---|---|---|
| Baseline | | | |
| Activation | | | |
| Optimizer | | | |
| Final best | | | |
