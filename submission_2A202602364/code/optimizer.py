"""Xay dung optimizer, scheduler va gradient clipping."""

from __future__ import annotations

import torch

OPTIMIZERS = (
    "sgd",
    "sgd_momentum",
    "adam",
    "adamw",
)

SCHEDULERS = (
    "cosine",
    "step",
)


def build_optimizer(
    name: str,
    params,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    betas=(0.9, 0.999),
    eps: float = 1e-8,
):
    """Tao optimizer theo ten va cau hinh."""
    if name not in OPTIMIZERS:
        raise ValueError(
            f"Optimizer '{name}' khong hop le. " f"Chon mot trong {OPTIMIZERS}."
        )

    if lr <= 0.0:
        raise ValueError("lr phai lon hon 0")

    if weight_decay < 0.0:
        raise ValueError("weight_decay khong duoc am")

    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum phai nam trong khoang [0, 1)")

    common_args = {
        "params": params,
        "lr": lr,
        "weight_decay": weight_decay,
    }

    if name == "sgd":
        return torch.optim.SGD(
            **common_args,
        )

    if name == "sgd_momentum":
        return torch.optim.SGD(
            **common_args,
            momentum=momentum,
        )

    if name == "adam":
        return torch.optim.Adam(
            **common_args,
            betas=betas,
            eps=eps,
        )

    return torch.optim.AdamW(
        **common_args,
        betas=betas,
        eps=eps,
    )


def build_scheduler(
    optimizer,
    name: str | None,
    total_steps: int,
    **kwargs,
):
    """Tao scheduler tuy chon."""
    if name is None or name == "none":
        return None

    if name not in SCHEDULERS:
        raise ValueError(
            f"Scheduler '{name}' khong hop le. " f"Chon mot trong {SCHEDULERS}."
        )

    if total_steps <= 0:
        raise ValueError("total_steps phai lon hon 0")

    if name == "cosine":
        eta_min = kwargs.get("eta_min", 0.0)

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=eta_min,
        )

    step_size = kwargs.get(
        "step_size",
        max(1, total_steps // 3),
    )
    gamma = kwargs.get("gamma", 0.1)

    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )


def clip_gradients(
    params,
    max_norm: float | None,
) -> float:
    """Tra ve global L2 gradient norm truoc khi cat."""
    parameters = [parameter for parameter in params if parameter.grad is not None]

    if not parameters:
        return 0.0

    if max_norm is not None and max_norm <= 0.0:
        raise ValueError("max_norm phai lon hon 0 hoac la None")

    clipping_threshold = float("inf") if max_norm is None else max_norm

    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=clipping_threshold,
        norm_type=2.0,
    )

    return total_norm.item()
