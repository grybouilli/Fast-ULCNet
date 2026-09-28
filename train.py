"""
train_fast_ulcnet.py
====================
PyTorch training script for Fast-ULCNet:
  "Fast-ULCNet: A Fast and Ultra Low Complexity Network for
   Single-Channel Speech Enhancement"
  Arrieta Larraza & de Koeijer, ICASSP 2025
  https://arxiv.org/abs/2601.14925
  https://github.com/narrietal/Fast-ULCNet

Architecture summary (from the paper):
  Stage 1 – Magnitude mask estimation
    - Power-law compression on STFT real & imaginary parts
    - Channel-wise Feature Reorientation (CFR) block
    - Conv Block: 4 depthwise-separable Conv1d layers (freq axis)
        filters: 32 → 64 → 96 → 128, kernel 1×3
        MaxPool(2) after layers 2-4
    - Bidirectional Freq-FastGRNN (64 units) + pointwise Conv (64 filters)
    - 2 × Subband Temporal FastGRNN blocks (each = 2 FastGRNN layers, 128 units)
    - 2 FC layers (257 neurons each) → real-valued magnitude mask

  Stage 2 – Phase refinement
    - 2 × Conv2d(32 filters, 1×3) on [estimated mag, noisy phase]
    - Pointwise Conv2d(2 channels) → complex ratio mask (CRM)
    - Complex ratio masking to reconstruct enhanced complex STFT

FastGRNN / Comfi-FastGRNN:
  z_t  = σ(W x_t + U h_{t-1} + b_z)
  h̃_t  = tanh(W x_t + U h_{t-1} + b_h)
  h_t  = (ζ(1 - z_t) + ν) ⊙ h̃_t + z_t ⊙ h_{t-1}
  Comfi extension:
  h_t_comfi = γ · h_t + (1-γ) · λ   (γ, λ trainable scalars)

Training setup (from paper §3.1.4):
  Dataset  : DNS Challenge 2020 (1000 h, 16 kHz, SNR ∈ [-10, 30] dB)
  STFT     : 32 ms window, 16 ms hop, 512-point FFT  → 257 freq bins
  Samples  : 10 s clips (160,000 samples @ 16 kHz)
  Batch    : 32, 4000 train steps/epoch, 1000 val steps/epoch
  Optimizer: Adam, lr=1e-3, gradient clip=3.0
  Scheduler: ReduceLROnPlateau (factor=0.5, patience=3 epochs)
  Early stop: patience=5 epochs (best val-loss checkpoint kept)

Loss function (eq. 5):
  L = (1/TF) Σ_t Σ_f  ( |S| - |Ŝ| |  +  | S - Ŝ | )
  i.e. L1 on magnitude + L1 on complex spectrogram.
"""

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, "./fast_ulcnet_networks/pytorch_version/")
from fast_ulcnet_networks.pytorch_version.FastULCNet import FastULCNet, STFTLayer, ISTFT
from dataset.dns_dataset import make_dns_loaders
from dataset.voicebankdemand_dataset import make_vbd_loaders, make_loaders
from losses import FastULCNetLoss, MSELoss, LossConfig
from optimizer_scheduler_inst import OptimizerSchedulerInst
import libsegmenter

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Training & Validation loops
# ===========================================================================
def progress_bar(current: int, total: int, width=40):
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    print(
        f"\r[{bar}] {percent:.0%} ({current}/{total} Batch)",
        end="",
        flush=True,
    )


def infer_one(
    model,
    clean,
    noisy,
    device,
    criterion,
    optimizer=None,
    is_input_temporal: bool = True,
    stft=None,
):
    if not is_input_temporal:
        noisy = stft(noisy)  # (B, T, F) complex
        clean = stft(clean)
    noisy = noisy.to(device)
    clean = clean.to(device)
    if optimizer != None:
        optimizer.zero_grad()
    pred = model(noisy)
    loss = criterion(pred, clean)
    return loss


def train_one_batch(
    model,
    clean,
    noisy,
    criterion,
    optimizer,
    device,
    grad_clip: float = 3.0,
    is_input_temporal=True,
    stft=None,
):
    loss = infer_one(
        model,
        clean,
        noisy,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        is_input_temporal=is_input_temporal,
        stft=stft,
    )
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    max_steps: int,
    grad_clip: float = 3.0,
    is_input_temporal=True,
    stft=None,
):
    model.train()
    total_loss = 0.0
    steps = 0
    batch_amount = len(loader)
    for batch_idx, (noisy, clean) in enumerate(loader):
        progress_bar(batch_idx + 1, batch_amount)
        if steps >= max_steps:
            break
        loss = train_one_batch(
            model,
            clean,
            noisy,
            criterion,
            optimizer,
            device,
            grad_clip,
            is_input_temporal,
            stft,
        )
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def validate(
    model, loader, criterion, device, max_steps: int, is_input_temporal=True, stft=None
):
    model.eval()
    total_loss = 0.0
    steps = 0
    batch_amount = len(loader)
    for batch_idx, (noisy, clean) in enumerate(loader):
        progress_bar(batch_idx + 1, batch_amount)
        if steps >= max_steps:
            break
        loss = infer_one(
            model,
            clean,
            noisy,
            device,
            criterion=criterion,
            is_input_temporal=is_input_temporal,
            stft=stft,
        )
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


# ===========================================================================
# Main training entry-point
# ===========================================================================


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    import yaml

    config = None
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    model = FastULCNet(config).to(device)

    training_parameters = config["training_parameters"]
    batch_size = training_parameters["batch_size"]
    max_epochs = training_parameters["max_epochs"]
    clip_len = training_parameters["clip_len"]
    early_stop_patience = training_parameters["early_stopping"]["patience"]
    is_input_temporal = config["model_parameters"]["temporal_input"]
    stft = None
    window = torch.from_numpy(
        libsegmenter.WindowSelector("hann75", "wola", model.win_size).analysis_window,
    ).float()
    if not is_input_temporal:
        stft = STFTLayer(model.n_fft, model.hop_size, model.win_size, window)

    train_loader, val_loader = None, None
    if args.dataset_dir == None:
        if args.vbd_ds == "Laroche":
            print("Using Voice-Bank-DEMAND-16k dataset with Laroche's implementation")
            train_loader, val_loader = (
                make_vbd_loaders(  # Voice-Bank-DEMAND-16k dataset
                    batch_size=batch_size,
                    clip_len=clip_len,
                    num_workers=args.num_workers,
                )
            )
        else:
            print("Using Voice-Bank-DEMAND-16k dataset with custom implementation")
            train_loader, val_loader = make_loaders(  # Voice-Bank-DEMAND-16k dataset
                batch_size=batch_size,
                clip_len=clip_len,
                mode="crop",
                num_workers=args.num_workers,
            )
    else:
        print(f"Using DNS Challenge 2020 dataset: {args.dataset_dir}")
        train_loader, val_loader = make_dns_loaders(
            args.dataset_dir, batch_size=batch_size, clip_len=clip_len
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.3f} M")

    # ------------------------------------------------------------------
    # Optimizer, scheduler & loss
    # ------------------------------------------------------------------
    osi = OptimizerSchedulerInst(config, model)
    optimizer = osi.optimizer

    loss_config = LossConfig(
        config,
        window,
        device,
    )
    criterion = loss_config.criterion

    # ------------------------------------------------------------------
    # (Optional) Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    best_state_dict = None
    no_improve_epochs = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, max_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_steps=args.train_steps,
            grad_clip=args.grad_clip,
            is_input_temporal=is_input_temporal,
            stft=stft,
        )
        print()
        val_loss = validate(
            model,
            val_loader,
            criterion,
            device,
            max_steps=args.val_steps,
            is_input_temporal=is_input_temporal,
            stft=stft,
        )
        osi.step(metric=val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"lr={current_lr:.2e}"
        )

        # Save checkpoint every epoch
        ckpt_path = os.path.join(args.save_dir, f"fast_ulcnet_epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "train_seq_len": clip_len,
            },
            ckpt_path,
        )

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve_epochs = 0
            torch.save(best_state_dict, os.path.join(args.save_dir, args.output_name))
            print(f"  ✓ New best model saved  (val_loss={best_val_loss:.5f})")
        else:
            no_improve_epochs += 1
            print(f"  No improvement for {no_improve_epochs} epoch(s).")
            if no_improve_epochs >= early_stop_patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.5f}")


# ===========================================================================
# CLI
# ===========================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Fast-ULCNet (Arrieta Larraza & de Koeijer, ICASSP 2025)"
    )
    # Paths
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Directory with noisy/ and clean/ sub-folders",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Where to save model checkpoints"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to a checkpoint to resume from"
    )

    # Model
    parser.add_argument(
        "--config",
        type=str,
        default="fast_ulcnet_networks/config.yml",
        help="Config file for model (.yaml)",
    )

    # Training hyperparameters (paper §3.1.4)
    parser.add_argument(
        "--train_steps",
        type=int,
        default=4000,
        help="Training steps per epoch (paper: 4000)",
    )
    parser.add_argument(
        "--val_steps",
        type=int,
        default=1000,
        help="Validation steps per epoch (paper: 1000)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=3.0,
        help="Gradient clipping norm (paper: 3.0)",
    )
    parser.add_argument("--output_name", type=str, default="fast_ulcnet_best.pt")

    # Misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vbd_ds", type=str, default="custom")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
