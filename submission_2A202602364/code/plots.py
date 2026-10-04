"""Ve bieu do cho tung thi nghiem va nhom thi nghiem."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Ve loss, validation metrics va gradient norm."""
    cfg = result["cfg"]
    history = result["history"]
    summary = result["summary"]

    epochs = history["epoch"]

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.5),
    )

    if not epochs:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "Diverged before first completed epoch",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()

        figure.suptitle(f"{cfg['exp_id']} | no completed epoch")
        figure.tight_layout()
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        print("Saved divergence figure:", output_path)
        return

    axes[0].plot(
        epochs,
        history["train_loss"],
        marker="o",
        markersize=3,
        label="Train loss",
    )
    axes[0].plot(
        epochs,
        history["val_loss"],
        marker="o",
        markersize=3,
        label="Validation loss",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        epochs,
        history["val_acc"],
        marker="o",
        markersize=3,
        label="Validation accuracy",
    )
    axes[1].plot(
        epochs,
        history["val_macro_f1"],
        marker="o",
        markersize=3,
        label="Validation macro-F1",
    )
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        epochs,
        history["grad_norm"],
        marker="o",
        markersize=3,
        color="tab:red",
        label="Gradient norm",
    )
    axes[2].set_title("Gradient norm before clipping")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Global L2 norm")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    best_epoch = summary["best_epoch"]

    if best_epoch > 0:
        for axis in axes:
            axis.axvline(
                best_epoch,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.65,
                label="Best epoch",
            )

    title = (
        f"{cfg['exp_id']} | "
        f"{cfg['optimizer']} | "
        f"lr={cfg['lr']} | "
        f"batch={cfg['batch']} | "
        f"{cfg['precision']} | "
        f"init={cfg['init']}"
    )

    figure.suptitle(title)
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved figure:", output_path)


def plot_compare(
    results: list[dict],
    metric: str,
    path: str,
    title: str = "",
) -> None:
    """Ve chong mot metric cua nhieu thi nghiem."""
    if not results:
        raise ValueError("results khong duoc rong")

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    for result in results:
        history = result["history"]

        if metric not in history:
            raise KeyError(f"Khong co metric '{metric}'")

        axis.plot(
            history["epoch"],
            history[metric],
            marker="o",
            markersize=3,
            label=result["cfg"]["exp_id"],
        )

    axis.set_title(title or f"Comparison: {metric}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric)
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved comparison figure:", output_path)
