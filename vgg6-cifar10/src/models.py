"""VGG6 (4 conv + 2 FC layers) with configurable activation and optional BatchNorm."""
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
}


def _act(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Options: {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


def _conv_block(in_c, out_c, act, batch_norm):
    layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=not batch_norm)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(_act(act))
    return layers


class VGG6(nn.Module):
    """conv64-pool-conv128-pool-conv256-conv256-pool-FC512-FC10  (6 weight layers)."""

    def __init__(self, activation: str = "relu", batch_norm: bool = True,
                 num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        layers = []
        layers += _conv_block(3, 64, activation, batch_norm)
        layers.append(nn.MaxPool2d(2))                      # 32 -> 16
        layers += _conv_block(64, 128, activation, batch_norm)
        layers.append(nn.MaxPool2d(2))                      # 16 -> 8
        layers += _conv_block(128, 256, activation, batch_norm)
        layers += _conv_block(256, 256, activation, batch_norm)
        layers.append(nn.MaxPool2d(2))                      # 8 -> 4
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            _act(activation),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_weights(activation)

    def _init_weights(self, activation):
        # Kaiming for ReLU-family, Xavier for saturating activations.
        relu_family = {"relu", "leaky_relu", "silu", "gelu", "selu"}
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if activation in relu_family:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))
