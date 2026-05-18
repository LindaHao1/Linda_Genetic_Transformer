"""
Train a deep ensemble surrogate world model on a processed simulation dataset.

Each ensemble member is a deterministic CNN trained independently with a different
random seed. Epistemic uncertainty is estimated from inter-member disagreement at
inference time — high variance across members indicates an OOD design.

Expects a processed dataset directory produced by process_dataset.py, which
contains pre-encoded image tensors (images.pt), S21 targets (s21.pt), and
metadata (meta.json). Use process_dataset.py to convert raw JSONL datasets first.

Usage:
    python world_model_test/train.py \
        --processed-dir /path/to/processed \
        --epochs 50 --batch-size 64 --ensemble-size 5 --num-workers 8 \
        --output-dir /tmp/wm_run
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent))

from model import EnsembleModel, WorldModel
from wm_dataset import ProcessedWMDataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ensemble surrogate world model")
    p.add_argument(
        "--processed-dir", required=True, metavar="DIR",
        help="Processed dataset directory produced by process_dataset.py "
             "(must contain images.pt, s21.pt, meta.json).",
    )
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--val-frac",      type=float, default=0.1)
    p.add_argument("--ensemble-size", type=int,   default=5,
                   help="Number of independently-trained ensemble members")
    p.add_argument("--output-dir",    default="/tmp/wm_run")
    p.add_argument("--device",        default=None)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42,
                   help="Base seed; member i uses seed+i for independent initialisation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-member training
# ---------------------------------------------------------------------------

def _run_epoch(
    model: WorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one train or val epoch. Returns (mean_mse, mean_mae)."""
    training = optimizer is not None
    model.train(training)
    total_mse = 0.0
    total_mae = 0.0
    n = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, s21_true in loader:
            images = images.to(device)
            s21_true = s21_true.to(device)

            if training:
                optimizer.zero_grad()
                loss = model.train_step(images, s21_true)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                loss = model.train_step(images, s21_true)

            pred = model(images)
            mae = (pred - s21_true).abs().mean().item()

            bs = images.size(0)
            total_mse += loss.item() * bs
            total_mae += mae * bs
            n += bs

    return total_mse / n, total_mae / n


def _train_member(
    member_idx: int,
    n_freq: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> tuple[WorldModel, float]:
    """Train one ensemble member. Returns (best_model_state, best_val_mae)."""
    torch.manual_seed(seed)
    model = WorldModel(n_freq=n_freq).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log_path = out_dir / f"training_log_member{member_idx}.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "val_mse", "train_mae_db", "val_mae_db", "lr"])

    best_val_mae = float("inf")
    best_state: dict = {}

    for epoch in range(1, epochs + 1):
        train_mse, train_mae = _run_epoch(model, train_loader, optimizer, device)
        val_mse, val_mae = _run_epoch(model, val_loader, None, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"  [Member {member_idx}] Epoch {epoch:3d}/{epochs}  "
            f"train MSE={train_mse:.4f}  val MSE={val_mse:.4f}  "
            f"train MAE={train_mae:.3f} dB  val MAE={val_mae:.3f} dB  "
            f"lr={lr_now:.2e}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_mse, val_mse, train_mae, val_mae, lr_now])

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val_mae


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_curves(out_dir: Path, n_members: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for i in range(n_members):
        log_path = out_dir / f"training_log_member{i}.csv"
        epochs, tr_mse, va_mse, tr_mae, va_mae = [], [], [], [], []
        with open(log_path) as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                tr_mse.append(float(row["train_mse"]))
                va_mse.append(float(row["val_mse"]))
                tr_mae.append(float(row["train_mae_db"]))
                va_mae.append(float(row["val_mae_db"]))

        alpha = 0.9 if i == 0 else 0.5
        ax1.plot(epochs, va_mse, alpha=alpha, label=f"member {i} val")
        ax2.plot(epochs, va_mae, alpha=alpha, label=f"member {i} val")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE (dB²)")
    ax1.set_title("Validation MSE Loss — all members")
    ax1.legend(fontsize=7)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (dB)")
    ax2.set_title("Validation S21 MAE — all members")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def _plot_predictions(
    ensemble: EnsembleModel,
    val_loader: DataLoader,
    freq_points: np.ndarray,
    out_dir: Path,
    device: torch.device,
    n_examples: int = 6,
) -> None:
    ensemble.eval()
    freq_ghz = freq_points / 1e9

    collected_images: list[torch.Tensor] = []
    collected_true: list[torch.Tensor] = []
    with torch.no_grad():
        for images, s21_true in val_loader:
            collected_images.append(images)
            collected_true.append(s21_true)
            if sum(x.size(0) for x in collected_images) >= n_examples:
                break

    images = torch.cat(collected_images)[:n_examples].to(device)
    s21_true = torch.cat(collected_true)[:n_examples].numpy()

    with torch.no_grad():
        mean, epistemic_var = ensemble.predict(images)
    mean = mean.cpu().numpy()
    epistemic_std = epistemic_var.sqrt().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        if i >= n_examples:
            ax.set_visible(False)
            continue
        ax.plot(freq_ghz, s21_true[i], "b-", linewidth=1.5, label="Ground truth")
        ax.plot(freq_ghz, mean[i], "r--", linewidth=1.5, label="Ensemble mean")
        ax.fill_between(
            freq_ghz,
            mean[i] - epistemic_std[i],
            mean[i] + epistemic_std[i],
            alpha=0.25,
            color="red",
            label="±1σ epistemic",
        )
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("S21 (dB)")
        ax.set_title(f"Sample {i + 1}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Ensemble predictions vs ground truth — shaded region = inter-member std (epistemic uncertainty)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "val_predictions.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    dataset = ProcessedWMDataset(args.processed_dir)
    meta = dataset.meta
    n_freq = dataset.n_freq
    freq_points = dataset.freq_points
    print(f"Processed dir : {args.processed_dir}")
    print(f"Samples       : {len(dataset)}")
    print(f"Image shape   : {tuple(dataset.images.shape[1:])}")
    print(f"Frequency     : {n_freq} pts, "
          f"{meta['freq_min']/1e9:.0f}–{meta['freq_max']/1e9:.0f} GHz")
    print(f"Source dirs   : {meta.get('source_dirs', [])}")
    print(f"Ensemble size : {args.ensemble_size}")

    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {n_train}, Val: {n_val}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Train each member with a different seed for independent initialisation
    members: list[WorldModel] = []
    member_maes: list[float] = []

    for i in range(args.ensemble_size):
        print(f"\n--- Training member {i} / {args.ensemble_size - 1} ---")
        member, best_mae = _train_member(
            member_idx=i,
            n_freq=n_freq,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed + i,
            device=device,
            out_dir=out_dir,
        )
        members.append(member)
        member_maes.append(best_mae)
        print(f"  Member {i} best val MAE: {best_mae:.3f} dB")

    # Save ensemble checkpoint
    ensemble = EnsembleModel(members)
    torch.save(
        {
            "member_states":  [m.state_dict() for m in members],
            "member_val_maes": member_maes,
            "n_freq":          n_freq,
            "freq_min":        meta["freq_min"],
            "freq_max":        meta["freq_max"],
            "processed_dir":   str(args.processed_dir),
            "ensemble_size":   args.ensemble_size,
        },
        out_dir / "ensemble_model.pt",
    )

    # Plots
    _plot_curves(out_dir, args.ensemble_size)
    print("\nSaved training_curves.png")

    ensemble.to(device).eval()
    _plot_predictions(ensemble, val_loader, freq_points, out_dir, device)
    print("Saved val_predictions.png")

    # Summary statistics using ensemble mean
    all_mae, all_rmse, all_err = [], [], []
    with torch.no_grad():
        for images, s21_true in val_loader:
            images = images.to(device)
            mean_pred, _ = ensemble.predict(images)
            err = (mean_pred.cpu() - s21_true).abs()
            all_mae.extend(err.mean(dim=-1).tolist())
            all_rmse.extend(err.pow(2).mean(dim=-1).sqrt().tolist())
            all_err.extend(err.flatten().tolist())

    print("\n=== Validation Summary (ensemble mean) ===")
    print(f"  MAE:              {np.mean(all_mae):.3f} dB")
    print(f"  RMSE:             {np.mean(all_rmse):.3f} dB")
    print(f"  90th-pct error:   {np.percentile(all_err, 90):.3f} dB")
    print(f"  Per-member MAEs:  {[round(m, 3) for m in member_maes]}")
    print(f"  Output dir:       {out_dir}")

    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(
            {
                "ensemble_val_mae_db": float(np.mean(all_mae)),
                "ensemble_val_rmse_db": float(np.mean(all_rmse)),
                "ensemble_val_p90_err_db": float(np.percentile(all_err, 90)),
                "member_val_maes": member_maes,
                "ensemble_size": args.ensemble_size,
                "epochs": args.epochs,
                "n_train": n_train,
                "n_val": n_val,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
