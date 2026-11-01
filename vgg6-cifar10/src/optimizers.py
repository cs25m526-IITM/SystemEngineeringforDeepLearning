"""Optimizer factory covering SGD, Nesterov-SGD, Adam, Adagrad, RMSprop and Nadam."""
import torch.optim as optim


def build_optimizer(name: str, params, lr: float, weight_decay: float, momentum: float):
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nesterov":
        # Nesterov requires momentum > 0.
        return optim.SGD(params, lr=lr, momentum=momentum if momentum > 0 else 0.9,
                         nesterov=True, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'")
