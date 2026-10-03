"""Dinh nghia mang MLP cho bai toan Forest CoverType."""

from __future__ import annotations

import torch
import torch.nn as nn

EXPECTED_PARAMS = {
    (256, 128): 47_879,
    (512, 256): 161_287,
    (256, 128, 64): 55_687,
}


class MLP(nn.Module):
    """MLP voi ReLU va dropout sau moi lop an."""

    def __init__(
        self,
        hidden=(256, 128),
        dropout: float = 0.0,
        init: str = "he",
        in_features: int = 54,
        num_classes: int = 7,
    ):
        super().__init__()

        hidden = tuple(hidden)

        if not hidden:
            raise ValueError("hidden khong duoc rong")

        if any(not isinstance(width, int) or width <= 0 for width in hidden):
            raise ValueError("hidden phai chua cac so nguyen duong")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout phai nam trong khoang [0, 1)")

        layers: list[nn.Module] = []
        width_in = in_features

        for width_out in hidden:
            layers.append(nn.Linear(width_in, width_out, bias=True))
            layers.append(nn.ReLU())

            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

            width_in = width_out

        layers.append(nn.Linear(width_in, num_classes, bias=True))

        self.net = nn.Sequential(*layers)
        init_weights(self, init)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Nhan x (B, 54), tra ve logits (B, 7)."""
        return self.net(x)


def init_weights(
    model: nn.Module,
    init: str,
) -> None:
    """Khoi tao trong so cua moi nn.Linear."""
    valid_initializers = {
        "zeros",
        "normal",
        "xavier",
        "he",
        "default",
    }

    if init not in valid_initializers:
        raise ValueError(f"init phai la mot trong " f"{sorted(valid_initializers)}")

    if init == "default":
        return

    for layer in model.modules():
        if not isinstance(layer, nn.Linear):
            continue

        if init == "zeros":
            nn.init.zeros_(layer.weight)

        elif init == "normal":
            nn.init.normal_(
                layer.weight,
                mean=0.0,
                std=0.01,
            )

        elif init == "xavier":
            nn.init.xavier_normal_(layer.weight)

        elif init == "he":
            nn.init.kaiming_normal_(
                layer.weight,
                nonlinearity="relu",
            )

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def count_params(model: nn.Module) -> int:
    """Dem tong so tham so co the huan luyen."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


@torch.no_grad()
def activation_stats(
    model: nn.Module,
    x: torch.Tensor,
) -> list[float]:
    """Do std cua kich hoat sau tung lop Linear."""
    was_training = model.training
    model.eval()

    h = x
    stats: list[float] = []

    for layer in model.net:
        h = layer(h)

        if isinstance(layer, nn.Linear):
            stats.append(h.std(unbiased=False).item())

    model.train(was_training)
    return stats
