"""Pipeline huan luyen va danh gia dung chung cho moi thi nghiem."""

from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import (
    build_optimizer,
    build_scheduler,
    clip_gradients,
)

DEFAULT_CFG = dict(
    exp_id="base-s1",
    group="baseline",
    description="Baseline M-base",
    loss="ce",
    optimizer="sgd_momentum",
    lr=None,
    weight_decay=0.0,
    momentum=0.9,
    batch=512,
    epochs=20,
    hidden=(256, 128),
    dropout=0.0,
    init="he",
    clip_norm=None,
    precision="fp32",
    scheduler=None,
    seed=1,
)


def set_seed(seed: int) -> None:
    """Dat seed cho Python, NumPy va PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Tinh macro-F1 tu ma tran nham lan."""
    cm = np.asarray(cm)

    if cm.shape != (7, 7):
        raise ValueError("cm phai co shape (7, 7)")

    true_positive = np.diag(cm).astype(np.float64)
    false_positive = cm.sum(axis=0) - true_positive
    false_negative = cm.sum(axis=1) - true_positive

    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_positive) > 0,
    )

    recall = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_negative) > 0,
    )

    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )

    return float(f1.mean())


@torch.no_grad()
def predict(
    model,
    X,
    batch_size: int = 8192,
) -> torch.Tensor:
    """Du doan nhan theo tung batch."""
    if batch_size <= 0:
        raise ValueError("batch_size phai lon hon 0")

    model.eval()
    predictions = []

    for start in range(0, len(X), batch_size):
        xb = X[start : start + batch_size]
        logits = model(xb)
        predictions.append(logits.argmax(dim=1))

    if not predictions:
        return torch.empty(
            0,
            dtype=torch.int64,
            device=X.device,
        )

    return torch.cat(predictions)


def compute_loss(
    logits,
    y,
    loss_name: str,
):
    """Tinh cross-entropy hoac MSE voi one-hot label."""
    if loss_name == "ce":
        return F.cross_entropy(logits, y)

    if loss_name == "mse":
        targets = F.one_hot(
            y,
            num_classes=logits.shape[1],
        ).to(dtype=logits.dtype)

        return F.mse_loss(
            logits,
            targets,
            reduction="mean",
        )

    raise ValueError("loss_name phai la 'ce' hoac 'mse'")


@torch.no_grad()
def evaluate(
    model,
    X,
    y,
    loss_name: str = "ce",
    batch_size: int = 8192,
) -> dict:
    """Tinh loss, accuracy va macro-F1."""
    if len(X) != len(y):
        raise ValueError("X va y phai co cung so mau")

    model.eval()

    total_loss = 0.0
    total_correct = 0
    confusion = torch.zeros(
        (7, 7),
        dtype=torch.int64,
        device=X.device,
    )

    for start in range(0, len(X), batch_size):
        xb = X[start : start + batch_size]
        yb = y[start : start + batch_size]

        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * len(yb)
        total_correct += (preds == yb).sum().item()

        flat_indices = yb * 7 + preds
        confusion += torch.bincount(
            flat_indices,
            minlength=49,
        ).reshape(7, 7)

    cm = confusion.cpu().numpy()

    return {
        "loss": total_loss / len(y),
        "acc": total_correct / len(y),
        "macro_f1": macro_f1_from_confusion(cm),
    }


def run_experiment(
    cfg: dict,
    data: dict,
) -> dict:
    """Huan luyen mot cau hinh va chon best epoch bang validation."""
    cfg = {**DEFAULT_CFG, **cfg}

    if cfg["lr"] is None:
        raise ValueError("Can dat lr truoc khi huan luyen")

    if cfg["epochs"] <= 0 or cfg["batch"] <= 0:
        raise ValueError("epochs va batch phai lon hon 0")

    precision = cfg["precision"]

    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision phai la fp32, fp16 hoac bf16")

    X_tr = data["X_tr"]
    y_tr = data["y_tr"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    device = X_tr.device
    set_seed(cfg["seed"])

    model = MLP(
        hidden=tuple(cfg["hidden"]),
        dropout=cfg["dropout"],
        init=cfg["init"],
    ).to(device)

    hidden = tuple(cfg["hidden"])

    if hidden in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden]

    optimizer = build_optimizer(
        name=cfg["optimizer"],
        params=model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        momentum=cfg["momentum"],
    )

    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])

    scheduler = build_scheduler(
        optimizer,
        name=cfg.get("scheduler"),
        total_steps=steps_per_epoch * cfg["epochs"],
    )

    use_amp = precision != "fp32"
    use_scaler = precision == "fp16"

    if use_amp and device.type != "cuda":
        raise ValueError("Mixed precision trong lab nay can CUDA")

    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU hien tai khong ho tro bf16")

    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg["seed"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step0_metrics = evaluate(
        model,
        X_val,
        y_val,
        loss_name=cfg["loss"],
    )

    print(f"{cfg['exp_id']} step-0 loss: " f"{step0_metrics['loss']:.4f}")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = step0_metrics["loss"]
    best_epoch = 0
    best_val_acc = step0_metrics["acc"]
    best_val_macro_f1 = step0_metrics["macro_f1"]

    best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    diverged = False

    for epoch in range(1, cfg["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_start = time.perf_counter()
        model.train()
        batch_grad_norms = []

        for xb, yb in iterate_batches(
            X_tr,
            y_tr,
            batch_size=cfg["batch"],
            generator=generator,
            shuffle=True,
        ):
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(xb)
                loss = compute_loss(
                    logits,
                    yb,
                    cfg["loss"],
                )

            if not torch.isfinite(loss).item():
                diverged = True
                break

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            grad_norm = clip_gradients(
                model.parameters(),
                cfg["clip_norm"],
            )

            if not math.isfinite(grad_norm):
                diverged = True
                break

            batch_grad_norms.append(grad_norm)

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

        if diverged:
            print(f"{cfg['exp_id']} diverged " f"tai epoch {epoch}")
            break

        train_metrics = evaluate(
            model,
            X_tr,
            y_tr,
            loss_name=cfg["loss"],
        )

        val_metrics = evaluate(
            model,
            X_val,
            y_val,
            loss_name=cfg["loss"],
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_time = time.perf_counter() - epoch_start
        mean_grad_norm = float(np.mean(batch_grad_norms))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(mean_grad_norm)
        history["epoch_time_s"].append(epoch_time)

        print(
            f"{cfg['exp_id']} "
            f"epoch {epoch:02d}/{cfg['epochs']} | "
            f"train {train_metrics['loss']:.4f} | "
            f"val {val_metrics['loss']:.4f} | "
            f"acc {val_metrics['acc']:.4f} | "
            f"macro-F1 {val_metrics['macro_f1']:.4f} | "
            f"grad {mean_grad_norm:.4f} | "
            f"{epoch_time:.2f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_acc = val_metrics["acc"]
            best_val_macro_f1 = val_metrics["macro_f1"]

            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    peak_mem_mb = 0.0

    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    if history["epoch"]:
        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        time_per_epoch = float(np.mean(history["epoch_time_s"]))
    else:
        final_train_loss = float("nan")
        final_val_loss = float("nan")
        time_per_epoch = float("nan")

    summary = {
        "step0_loss": step0_metrics["loss"],
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "val_acc": best_val_acc,
        "val_macro_f1": best_val_macro_f1,
        "time_per_epoch_s": time_per_epoch,
        "peak_mem_MB": peak_mem_mb,
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(
    row_id,
    preds,
    path: str,
) -> None:
    """Ghi CSV row_id,pred dung dinh dang cham diem."""
    row_id = np.asarray(row_id)
    preds = np.asarray(preds)

    if row_id.ndim != 1 or preds.ndim != 1:
        raise ValueError("row_id va preds phai la mang 1 chieu")

    if len(row_id) != len(preds):
        raise ValueError("row_id va preds phai co cung do dai")

    if len(np.unique(row_id)) != len(row_id):
        raise ValueError("row_id bi trung")

    if np.any((preds < 0) | (preds > 6)):
        raise ValueError("pred phai nam trong 0..6")

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.DataFrame(
        {
            "row_id": row_id.astype(np.int64),
            "pred": preds.astype(np.int64),
        }
    )

    frame.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved {len(frame)} predictions " f"to {output_path}")


def final_eval(
    cfg: dict,
    result: dict,
    data: dict,
    pred_path: str,
) -> None:
    """Du doan eval bang best_state va ghi file nop."""
    cfg = {**DEFAULT_CFG, **cfg}
    device = data["X_eval"].device

    model = MLP(
        hidden=tuple(cfg["hidden"]),
        dropout=cfg["dropout"],
        init=cfg["init"],
    ).to(device)

    model.load_state_dict(result["best_state"])
    model.eval()

    preds = predict(
        model,
        data["X_eval"],
    )

    write_predictions(
        data["eval_row_id"],
        preds.cpu().numpy(),
        pred_path,
    )
