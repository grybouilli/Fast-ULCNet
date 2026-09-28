from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import re
import torchaudio

# ===========================================================================
# Dataset  (DNS Challenge 2020 compatible)
# ===========================================================================


class DNSDataset(Dataset):
    """
    Expects a root directory containing two sub-folders:
        root/noisy/   – mixed speech files (*.wav)
        root/clean/   – corresponding clean speech files (*.wav)
    Files must be paired by name.

    Audio at 16 kHz, clips of exactly 10 s (160,000 samples).
    STFT: 32 ms window (512 samples), 16 ms hop (256 samples), 512-pt FFT.
    """

    SAMPLE_RATE = 16_000

    def __init__(self, root: str, clip_len: int | None = None):
        noisy_dir = Path(root) / "noisy"
        clean_dir = Path(root) / "clean"
        self.clip_len = clip_len

        noisy_files = sorted(noisy_dir.glob("*.wav"))
        filenums = [
            m.group(1)
            for f in noisy_files
            if (m := re.search(r"_fileid_(\d+)\.wav$", f.name))
        ]
        clean_files = [clean_dir / f"clean_fileid_{n}.wav" for n in filenums]
        noisy_files = [
            f
            for f, n in zip(noisy_files, filenums)
            if (clean_dir / f"clean_fileid_{n}.wav").exists()
        ]
        self.pairs = [(n, c) for n, c in zip(noisy_files, clean_files) if c.exists()]

    def __len__(self):
        return len(self.pairs)

    def _load(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))
        assert sr == self.SAMPLE_RATE, f"Expected 16 kHz, got {sr}"
        wav = wav.mean(0)  # mono
        # Pad or trim to exactly CLIP_LEN samples
        if wav.shape[-1] < self.clip_len:
            wav = F.pad(wav, (0, self.clip_len - wav.shape[-1]))
        elif wav.shape[-1] > self.clip_len:
            max_start = wav.shape[-1] - self.clip_len
            start = int(torch.randint(0, max_start + 1, (1,)).item())
            wav = wav[start : start + self.clip_len]
        return wav

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy_wav = self._load(noisy_path)
        clean_wav = self._load(clean_path)
        return clean_wav, noisy_wav


def make_dns_loaders(
    dataset_dir: str | Path,
    batch_size: int = 16,
    clip_len: int = 160_000,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    from torch.utils.data import DataLoader, random_split

    dataset = DNSDataset(
        root=dataset_dir,
        clip_len=clip_len,
    )
    train_ds, val_ds = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
