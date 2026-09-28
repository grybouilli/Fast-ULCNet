import numpy as np
import torch
import torch.nn as nn


class FastULCNetLoss(nn.Module):
    """
    Spectrogram-domain loss for waveform-to-waveform enhancement.

    The model operates on raw waveforms:

        pred   : (B, samples)
        target : (B, samples)

    Both are converted to complex STFTs before computing:

        L = L_mag + L_complex

    where

        L_mag     = mean(||S_pred| - |S_target||)
        L_complex = mean(|S_pred - S_target|)
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: torch.Tensor,
        is_input_temporal=True,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.is_input_temporal = is_input_temporal
        # Important: move this with the loss when calling loss.to(device)
        self.register_buffer("window", window)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred   : (B, samples) predicted waveform
            target : (B, samples) clean waveform

        Returns:
            Scalar spectrogram-domain loss.
        """
        pred_stft = pred
        target_stft = target
        if self.is_input_temporal:
            pred_stft = torch.stft(
                pred,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
            )

            target_stft = torch.stft(
                target,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
            )

        pred_mag = pred_stft.abs()
        target_mag = target_stft.abs()

        mag_loss = (pred_mag - target_mag).abs().mean()
        cplx_loss = (pred_stft - target_stft).abs().mean()

        return mag_loss + cplx_loss


import torch
import torch.nn as nn
import torch.nn.functional as F

# ── helpers ──────────────────────────────────────────────────────────────────


def stft(
    signal: torch.Tensor,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
) -> torch.Tensor:
    """Compute STFT and return complex spectrogram (batch, freq, time)."""
    window = torch.hann_window(
        n_fft if win_length is None else win_length, device=signal.device
    )
    return torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length or n_fft // 4,
        win_length=win_length or n_fft,
        window=window,
        return_complex=True,
    )


# ── Loss 1: MSE in the compressed frequency domain ───────────────────────────
class MSELoss(nn.Module):
    """
    L_MSE  –  Mean Squared Error in the compressed frequency domain.

    Following [13], magnitude compression is applied before computing MSE:
        L_MSE = || |S|^α  -  |Ŝ|^α ||²_F
    where α ∈ (0, 1] controls compression strength (default 0.3).
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        alpha: float = 0.3,
        is_input_temporal=True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha
        self.is_input_temporal = is_input_temporal

    def forward(
        self, s: torch.Tensor, s_hat: torch.Tensor, eps: float = 1e-5
    ) -> torch.Tensor:
        S = s
        S_hat = s_hat
        if self.is_input_temporal:
            S = stft(s, self.n_fft, self.hop_length)
            S_hat = stft(s_hat, self.n_fft, self.hop_length)

        mag = (S.abs() + eps).pow(self.alpha)
        mag_hat = (S_hat.abs() + eps).pow(self.alpha)

        return F.mse_loss(mag, mag_hat)


# Loss instanciater
class LossConfig:
    """
    Instantiates a loss function from a parsed YAML config dict.

    Expected config structure:
        data_parameters:
          n_fft: 512
          hop_size: 256
          win_size: 512
          ...
        model_parameters:
          temporal_input: False
        training_parameters:
          loss: MSE
          MSE:                  # optional block to override defaults
            alpha: 0.5
    """

    SUPPORTED_LOSSES = {"MSE", "FastULCNet"}

    def __init__(self, config: dict, window: torch.Tensor, device: torch.device):
        data_p = config.get("data_parameters", {})
        model_p = config.get("model_parameters", {})
        train_p = config.get("training_parameters", {})

        self.loss_name = train_p.get("loss")
        self.loss_overrides = train_p.get(self.loss_name, {})  # optional per-loss block
        self.is_temporal = model_p.get("temporal_input", False)
        self.n_fft = data_p.get("n_fft", 512)
        self.hop_length = data_p.get("hop_size", 256)
        self.win_length = data_p.get("win_size", 512)
        self.window = window
        self.device = device

        if self.loss_name not in self.SUPPORTED_LOSSES:
            raise ValueError(
                f"Unsupported loss '{self.loss_name}'. "
                f"Choose from: {self.SUPPORTED_LOSSES}"
            )

        self.criterion = self._build_loss()

    def _get(self, key, default):
        """Resolve a param: per-loss override block takes priority over the default."""
        return self.loss_overrides.get(key, default)

    def _build_loss(self) -> nn.Module:
        match self.loss_name:
            case "MSE":
                criterion = MSELoss(
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    alpha=self._get("alpha", 0.3),
                    is_input_temporal=self.is_temporal,
                )

            case "FastULCNet":
                criterion = FastULCNetLoss(
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=self.window,
                    is_input_temporal=self.is_temporal,
                )

        return criterion.to(self.device)

    def __call__(self, *args, **kwargs):
        """Delegate directly so LossConfig can be used in place of the criterion."""
        return self.criterion(*args, **kwargs)

    def __repr__(self):
        return (
            f"LossConfig(\n"
            f"  loss={self.loss_name},\n"
            f"  is_temporal={self.is_temporal},\n"
            f"  overrides={self.loss_overrides}\n"
            f")"
        )


# ── quick sanity-check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    SR = 16_000
    B, T = 2, SR  # 1-second batch of 2 signals

    clean = torch.randn(B, T)
    noisy = clean + 0.05 * torch.randn(B, T)

    mse_fn = MSELoss()  # needs sample_rate kwarg if added
    ms_fn = MultiScaleLoss(sample_rate=SR)
    mt_fn = MultiTargetLoss(sample_rate=SR)

    # MSELoss doesn't use sample_rate – pass n_fft directly
    mse_fn = MSELoss(n_fft=512, hop_length=128, alpha=0.3)

    l_mse = mse_fn(clean, noisy)
    l_ms, l_cs, l_spec = ms_fn(clean, noisy)
    l_mt, l_s, l_p = mt_fn(clean, noisy)

    print(f"L_MSE  = {l_mse:.4f}")
    print(f"L_MS   = {l_ms:.4f}  (CS={l_cs:.4f}, spec={l_spec:.4f})")
    print(f"L_MT   = {l_mt:.4f}  (spec={l_s:.4f}, phase={l_p:.4f})")
