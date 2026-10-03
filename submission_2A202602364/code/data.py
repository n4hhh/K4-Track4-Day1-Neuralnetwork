"""Nap, chia validation, chuan hoa va chia lo du lieu CoverType."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10


def load_split(processed_dir: str = "data/processed"):
    """Nap train va eval da duoc tao boi scripts/split_data.py."""
    processed_path = Path(processed_dir)
    train_path = processed_path / "train.npz"
    eval_path = processed_path / "eval.npz"

    if not train_path.is_file() or not eval_path.is_file():
        raise FileNotFoundError(
            "Khong tim thay train.npz/eval.npz. "
            "Hay chay python scripts/split_data.py tu thu muc goc repo."
        )

    with np.load(train_path, allow_pickle=False) as train_data:
        X_train_full = train_data["X"]
        y_train_full = train_data["y"]

    with np.load(eval_path, allow_pickle=False) as eval_data:
        X_eval = eval_data["X"]
        y_eval = eval_data["y"]
        eval_row_id = eval_data["row_id"]

    assert X_train_full.shape == (464_809, 54)
    assert y_train_full.shape == (464_809,)
    assert X_eval.shape == (116_203, 54)
    assert y_eval.shape == (116_203,)
    assert eval_row_id.shape == (116_203,)

    assert X_train_full.dtype == np.float32
    assert X_eval.dtype == np.float32
    assert y_train_full.dtype == np.int64
    assert y_eval.dtype == np.int64
    assert eval_row_id.dtype == np.int64

    assert y_train_full.min() == 0
    assert y_train_full.max() == 6
    assert y_eval.min() == 0
    assert y_eval.max() == 6

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(
    X,
    y,
    val_fraction: float = 0.2,
    seed: int = 42,
):
    """Tach validation tu train theo phan tang nhan."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction phai nam trong khoang (0, 1)")

    if len(X) != len(y):
        raise ValueError("X va y phai co cung so mau")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=val_fraction,
        random_state=seed,
        stratify=y,
    )

    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tinh mean/std cua 10 cot so chi tu phan train con lai."""
    if X_tr.ndim != 2 or X_tr.shape[1] != 54:
        raise ValueError("X_tr phai co shape (N, 54)")

    numeric = X_tr[:, :N_NUMERIC]

    mean = numeric.mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    std = numeric.std(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    std = np.where(
        std <= np.finfo(np.float32).eps,
        1.0,
        std,
    ).astype(np.float32)

    return mean, std


def apply_standardizer(X, mean, std):
    """Chuan hoa 10 cot dau; giu nguyen 44 cot one-hot."""
    if X.ndim != 2 or X.shape[1] != 54:
        raise ValueError("X phai co shape (N, 54)")

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if mean.shape != (N_NUMERIC,):
        raise ValueError("mean phai co shape (10,)")

    if std.shape != (N_NUMERIC,):
        raise ValueError("std phai co shape (10,)")

    safe_std = np.where(std == 0.0, 1.0, std)

    X_scaled = np.array(
        X,
        dtype=np.float32,
        copy=True,
    )

    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / safe_std

    return X_scaled


def prepare_data(
    device: str,
    val_fraction: float = 0.2,
    seed: int = 42,
    processed_dir: str = "data/processed",
) -> dict:
    """Chuan bi toan bo train, validation va eval tren device."""
    (
        X_train_full,
        y_train_full,
        X_eval,
        y_eval,
        eval_row_id,
    ) = load_split(processed_dir)

    X_tr, y_tr, X_val, y_val = make_val_split(
        X_train_full,
        y_train_full,
        val_fraction=val_fraction,
        seed=seed,
    )

    mean, std = fit_standardizer(X_tr)

    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    target_device = torch.device(device)

    data = {
        "X_tr": torch.from_numpy(X_tr).to(target_device),
        "y_tr": torch.from_numpy(y_tr).to(
            target_device,
            dtype=torch.int64,
        ),
        "X_val": torch.from_numpy(X_val).to(target_device),
        "y_val": torch.from_numpy(y_val).to(
            target_device,
            dtype=torch.int64,
        ),
        "X_eval": torch.from_numpy(X_eval).to(target_device),
        "y_eval": torch.from_numpy(y_eval).to(
            target_device,
            dtype=torch.int64,
        ),
        "eval_row_id": eval_row_id,
        "standardizer_mean": mean,
        "standardizer_std": std,
    }

    majority_class = torch.bincount(data["y_tr"]).argmax()
    majority_accuracy = (data["y_val"] == majority_class).float().mean().item()

    print(
        f"train: {tuple(data['X_tr'].shape)}, "
        f"val: {tuple(data['X_val'].shape)}, "
        f"eval: {tuple(data['X_eval'].shape)}"
    )
    print(
        f"Lop da so tren train: {majority_class.item()}, "
        f"accuracy tren val: {majority_accuracy:.4f}"
    )

    return data


def iterate_batches(
    X,
    y,
    batch_size: int,
    generator: torch.Generator | None = None,
    shuffle: bool = True,
):
    """Tra ve tung mini-batch; batch cuoi co the nho hon batch_size."""
    if len(X) != len(y):
        raise ValueError("X va y phai co cung so mau")

    if batch_size <= 0:
        raise ValueError("batch_size phai la so nguyen duong")

    if shuffle:
        permutation_device = X.device if generator is None else generator.device
        indices = torch.randperm(
            len(X),
            generator=generator,
            device=permutation_device,
        )
        if indices.device != X.device:
            indices = indices.to(X.device)
    else:
        indices = torch.arange(
            len(X),
            device=X.device,
        )

    for start in range(0, len(X), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield X[batch_indices], y[batch_indices]
