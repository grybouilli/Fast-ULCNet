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


def cosine_similarity_loss(s: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
    """Time-domain cosine-similarity loss: 1 - cos(s, s_hat)."""
    # s, s_hat: (batch, time)
    return 1.0 - F.cosine_similarity(s, s_hat, dim=-1).mean()


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


# ── Loss 2: Multi-Scale loss  (L_MS) ─────────────────────────────────────────


class MultiScaleLoss(nn.Module):
    """
    L_MS  =  Σ_j (1/K) Σ_{k=1}^{K} CS(s_{jk}, ŝ_{jk})
           + Σ_i || |S_i|^α  -  |Ŝ_i|^α ||²_F

    where
      • j ∈ L = {16, …, 128} ms indexes segment lengths used for
        the time-domain cosine-similarity (CS) term,
      • i ∈ {1, …, I} indexes STFT window sizes W = {16, …, 64} ms,
      • α controls magnitude compression (default 0.3).

    The spectral term is called L_spec in the paper.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        segment_lengths_ms: list[int] | None = None,  # j
        stft_window_sizes_ms: list[int] | None = None,  # i
        alpha: float = 0.3,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.alpha = alpha

        # segment lengths L for the CS term
        seg_ms = segment_lengths_ms or list(range(16, 129, 16))  # 16…128 ms
        self.segment_lengths = [int(ms * sample_rate / 1000) for ms in seg_ms]

        # STFT window sizes W for the spectral term
        win_ms = stft_window_sizes_ms or list(range(16, 65, 16))  # 16…64 ms
        self.stft_windows = [int(ms * sample_rate / 1000) for ms in win_ms]

    # ---- spectral term L_spec -----------------------------------------------
    def _l_spec(self, s: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=s.device)
        for win in self.stft_windows:
            S = stft(s, n_fft=win)
            S_hat = stft(s_hat, n_fft=win)
            mag = S.abs().pow(self.alpha)
            mag_hat = S_hat.abs().pow(self.alpha)
            # Frobenius norm squared, averaged over batch
            diff = mag - mag_hat
            loss = loss + (diff * diff).sum(dim=(-2, -1)).mean()
        return loss

    # ---- time-domain CS term ------------------------------------------------
    def _l_cs(self, s: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
        """
        For every segment length j, chop the signal into K non-overlapping
        frames and average the cosine-similarity losses.
        """
        loss = torch.tensor(0.0, device=s.device)
        T = s.shape[-1]
        for seg_len in self.segment_lengths:
            K = T // seg_len
            if K == 0:
                continue
            # (batch, K, seg_len)
            s_segs = s[..., : K * seg_len].reshape(s.shape[0], K, seg_len)
            s_hat_segs = s_hat[..., : K * seg_len].reshape(s_hat.shape[0], K, seg_len)
            # flatten batch & K for cosine_similarity
            s_flat = s_segs.reshape(-1, seg_len)
            s_hat_flat = s_hat_segs.reshape(-1, seg_len)
            cs = 1.0 - F.cosine_similarity(s_flat, s_hat_flat, dim=-1)
            loss = loss + cs.mean()
        return loss / max(len(self.segment_lengths), 1)

    def forward(self, s: torch.Tensor, s_hat: torch.Tensor):
        l_cs = self._l_cs(s, s_hat)
        l_spec = self._l_spec(s, s_hat)
        return l_cs + l_spec, l_cs, l_spec  # return components for logging


# ── Loss 3: Multi-Target loss  (L_MT) ────────────────────────────────────────


class MultiTargetLoss(nn.Module):
    """
    L_MT  =  L_spec  +  Σ_i || |S_i|^α ⊙ e^{jφ_s}  -  |Ŝ_i|^α ⊙ e^{jφ_ŝ} ||²_F

    Adds a phase-aware term on top of L_spec.  The Hadamard product ⊙ with
    e^{jφ} reconstructs a 'phase-aware compressed spectrogram', allowing the
    loss to penalise phase errors in addition to magnitude errors.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        stft_window_sizes_ms: list[int] | None = None,
        alpha: float = 0.3,
    ):
        super().__init__()
        self.alpha = alpha
        win_ms = stft_window_sizes_ms or list(range(16, 65, 16))
        self.stft_windows = [int(ms * sample_rate / 1000) for ms in win_ms]

    def forward(self, s: torch.Tensor, s_hat: torch.Tensor):
        l_spec = torch.tensor(0.0, device=s.device)
        l_phase = torch.tensor(0.0, device=s.device)

        for win in self.stft_windows:
            S = stft(s, n_fft=win)  # complex (B, F, T)
            S_hat = stft(s_hat, n_fft=win)

            mag = S.abs()
            mag_hat = S_hat.abs()

            # ---- L_spec term ------------------------------------------------
            diff_mag = mag.pow(self.alpha) - mag_hat.pow(self.alpha)
            l_spec = l_spec + (diff_mag * diff_mag).sum(dim=(-2, -1)).mean()

            # ---- Phase term -------------------------------------------------
            # e^{jφ} = S / |S|  (unit-magnitude phasor)
            eps = 1e-8
            phase_s = S / (mag + eps)  # complex, |·| = 1
            phase_s_hat = S_hat / (mag_hat + eps)

            # |S|^α ⊙ e^{jφ_s}  — compressed magnitude × phase phasor
            target = mag.pow(self.alpha) * phase_s  # complex product
            pred = mag_hat.pow(self.alpha) * phase_s_hat

            diff = target - pred
            # ||·||²_F  over (freq, time), averaged over batch
            l_phase = l_phase + (diff.abs() ** 2).sum(dim=(-2, -1)).mean()

        l_mt = l_spec + l_phase
        return l_mt, l_spec, l_phase  # return components for logging


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
