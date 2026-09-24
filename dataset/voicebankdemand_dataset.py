"""
voicebank_dataset.py
====================
PyTorch Dataset for JacobLinCool/VoiceBank-DEMAND-16k that produces
fixed-length STFT pairs suitable for Fast-ULCNet training.

Dataset schema
--------------
  id    : str
  clean : Audio(sampling_rate=16000)   → dict {"array": np.ndarray, "sampling_rate": int}
  noisy : Audio(sampling_rate=16000)

Clip-length strategy
--------------------
VoiceBank-DEMAND clips are real speech utterances (typically 1–10 s),
so their lengths vary and the naive options each have trade-offs:

  ❌ Discard short clips     – throws away a large fraction of the data
                               and biases training toward longer utterances.
  ❌ Always pad short clips  – introduces hard silence at a fixed boundary,
                               which the model will learn to handle but is
                               wasteful.
  ✅ Implemented here (default = "chunk"):
       • Clips longer than CLIP_LEN → split into as many non-overlapping
         full-length chunks as possible; the trailing remainder is DISCARDED
         (not padded), so every sample the model sees is exactly CLIP_LEN.
         Almost nothing is wasted on typical 2–10 s clips vs 2-second chunks.
       • Clips shorter than CLIP_LEN → by default PADDED with silence
         (pad_short=True).  Set pad_short=False to discard them instead.

  Alternative strategy (mode="crop"):
       • Clips longer than CLIP_LEN → ONE random crop per call to __getitem__
         (non-deterministic; good for augmentation).
       • Clips shorter than CLIP_LEN → same pad_short logic as above.
       This halves index-building overhead and gives different crop offsets
       each epoch, but is less sample-efficient.

Usage
-----
    from datasets import load_dataset
    from voicebank_dataset import VoiceBankDemandDataset

    hf_ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

    train_ds = VoiceBankDemandDataset(hf_ds["train"])
    val_ds   = VoiceBankDemandDataset(hf_ds["test"])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=4, pin_memory=True)

    for noisy_stft, clean_stft in train_loader:
        ...   # (B, T, F) complex64
"""

import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import hashlib
from pathlib import Path
import random

# ---------------------------------------------------------------------------
# STFT / audio constants  (must match the model's training setup)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
CLIP_LEN    = 32_000   # 2 s – a sensible default for VoiceBank-DEMAND;
                        # change to 160_000 for 10 s clips like DNS2020.
                        # Shorter clips → more variety per epoch, less
                        # memory per batch.
N_FFT       = 512
HOP_LENGTH  = 256       # 16 ms  (paper: 16 ms hop)
WIN_LENGTH  = 512       # 32 ms  (paper: 32 ms window)


def _to_mono_tensor(audio_dict: dict) -> torch.Tensor:
    """
    Convert a HuggingFace Audio dict → mono float32 torch.Tensor of shape (N,).
    The dict has keys 'array' (np.ndarray) and 'sampling_rate' (int).
    """
    arr = audio_dict["array"]
    wav = torch.from_numpy(arr.astype(np.float32))
    if wav.ndim == 2:               # (channels, samples)
        wav = wav.mean(dim=0)
    elif wav.ndim > 2:
        raise ValueError(f"Unexpected audio shape: {wav.shape}")
    return wav                      # (N,)


# ---------------------------------------------------------------------------
# Index builder helpers
# ---------------------------------------------------------------------------

def _build_chunk_index(hf_dataset, clip_len: int, pad_short: bool) -> list[tuple[int, int]]:
    """
    Pre-scan every example and record (example_idx, start_sample) pairs so
    that __getitem__ can directly extract the right fixed-length slice.

    Mode: chunk – non-overlapping full-length windows.
    Short clips (< clip_len):
        pad_short=True  → one entry at start=0  (will be zero-padded at load time)
        pad_short=False → skipped
    """
    index: list[tuple[int, int]] = []
    for i, example in enumerate(hf_dataset):
        n = len(example["noisy"]["array"])
        if n < clip_len:
            if pad_short:
                index.append((i, 0))
        else:
            n_chunks = n // clip_len          # integer divide → no remainder
            for c in range(n_chunks):
                index.append((i, c * clip_len))
    return index


def _build_crop_index(hf_dataset, clip_len: int, pad_short: bool) -> list[tuple[int, int]]:
    """
    Mode: crop – one entry per example, start=-1 means "crop randomly at runtime".
    Short clips get start=0 and will be padded.
    """
    index: list[tuple[int, int]] = []
    for i, example in enumerate(hf_dataset):
        n = len(example["noisy"]["array"])
        if n < clip_len:
            if pad_short:
                index.append((i, 0))          # 0 → will pad
        else:
            index.append((i, -1))             # -1 → random crop at __getitem__
    return index


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class VoiceBankDemandDataset(Dataset):
    """
    Wraps the JacobLinCool/VoiceBank-DEMAND-16k HuggingFace dataset and
    emits (noisy_stft, clean_stft) pairs as complex64 tensors of shape (T, F).

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        One split of the loaded HuggingFace dataset, e.g. ds["train"].
    clip_len : int
        Number of audio samples per training clip (default 32,000 = 2 s).
    mode : "chunk" | "crop"
        How to handle clips longer than clip_len.
        "chunk" (default) – non-overlapping windows, maximally sample-efficient.
        "crop"            – one random crop per example per epoch, more augmentation.
    pad_short : bool
        What to do with clips shorter than clip_len.
        True  (default) – zero-pad on the right.
        False           – discard the clip entirely.
    """

    def __init__(self, hf_dataset, clip_len=CLIP_LEN, window_len=WIN_LENGTH,
                 mode="chunk", pad_short=True, index_cache_dir="./index_cache"):
        self.ds        = hf_dataset
        self.clip_len  = clip_len
        self.mode      = mode
        self.pad_short = pad_short
        self.window    = torch.hann_window(window_len)

        cache_path = self._cache_path(index_cache_dir)

        if cache_path.exists():
            print(f"Loading cached index from {cache_path} ...")
            self._index = self._load_index(cache_path)
        else:
            print(f"Building index (mode='{mode}', clip_len={clip_len}, "
                  f"pad_short={pad_short}) over {len(hf_dataset)} examples ...")
            if mode == "chunk":
                self._index = _build_chunk_index(hf_dataset, clip_len, pad_short)
            elif mode == "crop":
                self._index = _build_crop_index(hf_dataset, clip_len, pad_short)
            else:
                raise ValueError(f"mode must be 'chunk' or 'crop', got '{mode}'")
            self._save_index(self._index, cache_path)
            print(f"Index cached to {cache_path}")

        n_orig = len(hf_dataset)
        n_idx  = len(self._index)
        print(f"Index ready: {n_idx} samples from {n_orig} examples "
              f"({n_idx / n_orig:.1f}× expansion).")

    def _cache_path(self, cache_dir: str) -> Path:
        """Unique filename per (dataset size, mode, clip_len, pad_short)."""
        key = f"{len(self.ds)}-{self.mode}-{self.clip_len}-{self.pad_short}"
        digest = hashlib.md5(key.encode()).hexdigest()[:10]
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return Path(cache_dir) / f"index_{digest}.json"

    @staticmethod
    def _save_index(index: list[tuple[int, int]], path: Path):
        with open(path, "w") as f:
            json.dump(index, f)

    @staticmethod
    def _load_index(path: Path) -> list[tuple[int, int]]:
        with open(path) as f:
            return [tuple(x) for x in json.load(f)]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        example_idx, start = self._index[idx]
        example = self.ds[example_idx]

        noisy_wav = _to_mono_tensor(example["noisy"])
        clean_wav = _to_mono_tensor(example["clean"])

        # ---- Determine crop/pad ----
        if start == -1:
            # Random crop (mode="crop", clip is long enough)
            max_start = len(noisy_wav) - self.clip_len
            start = int(torch.randint(0, max_start + 1, (1,)).item())

        noisy_wav = self._extract(noisy_wav, start)
        clean_wav = self._extract(clean_wav, start)

        # ---- STFT ----
        # noisy_stft = _stft(noisy_wav, self.window)   # (T, F) complex64
        # clean_stft = _stft(clean_wav, self.window)   # (T, F) complex64

        return noisy_wav, clean_wav

    # ------------------------------------------------------------------

    def _extract(self, wav: torch.Tensor, start: int) -> torch.Tensor:
        """Slice [start : start + clip_len], padding if necessary."""
        chunk = wav[start : start + self.clip_len]
        if chunk.shape[0] < self.clip_len:
            chunk = F.pad(chunk, (0, self.clip_len - chunk.shape[0]))
        return chunk

class VBDDataset(torch.utils.data.Dataset):
    """
    Copyright (c) 2023 Yexin Lu (MP-SENet)
    Copyright (c) 2025-2026 Clément Laroche
    Dataset from : https://github.com/LarocheC/eco8-neaixt
    
    Wraps a HuggingFace VoiceBank-DEMAND-16k split.

    Each row exposes paired ``clean`` and ``noisy`` audio decoded to numpy
    arrays at 16 kHz. The clean/noisy lengths always match by construction,
    so we crop a random ``segment_size`` window during training and return
    the full utterance during validation.
    """

    def __init__(self, hf_split, segment_size : int, split=True,
                 shuffle=True, seed=1234):
        self.hf_split = hf_split
        self.segment_size = segment_size
        self.split = split

        self.indices = list(range(len(hf_split)))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.hf_split[self.indices[index]]
        clean_audio = np.asarray(item["clean"]["array"], dtype=np.float32)
        noisy_audio = np.asarray(item["noisy"]["array"], dtype=np.float32)

        length = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:length]
        noisy_audio = noisy_audio[:length]

        clean_audio = torch.from_numpy(clean_audio)
        noisy_audio = torch.from_numpy(noisy_audio)

        norm_factor = torch.sqrt(len(noisy_audio) / (torch.sum(noisy_audio ** 2.0) + 1e-8))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start: audio_start + self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start + self.segment_size]
            else:
                pad = self.segment_size - clean_audio.size(1)
                clean_audio = torch.nn.functional.pad(clean_audio, (0, pad), 'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad), 'constant')

        return clean_audio.squeeze(0), noisy_audio.squeeze(0)

# ---------------------------------------------------------------------------
# collate_fn  – needed because complex tensors require explicit stacking
# ---------------------------------------------------------------------------

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """
    Stack a list of (noisy_stft, clean_stft) pairs into batched tensors.
    Returns:
        noisy : (B, T, F)  complex64
        clean : (B, T, F)  complex64
    """
    noisy_list, clean_list = zip(*batch)
    return torch.stack(noisy_list), torch.stack(clean_list)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_loaders(
    batch_size:  int  = 16,
    clip_len:    int  = CLIP_LEN,
    window_len:    int  = WIN_LENGTH,
    mode:        Literal["chunk", "crop"] = "chunk",
    pad_short:   bool = True,
    num_workers: int  = 4,
    pin_memory:  bool = True,
):
    """
    Load the HuggingFace dataset and return (train_loader, val_loader).

    Example
    -------
        train_loader, val_loader = make_loaders(batch_size=16, clip_len=32_000)
        for noisy_stft, clean_stft in train_loader:
            pred = model(noisy_stft.to(device))
            loss = criterion(pred, clean_stft.to(device))
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    print("Downloading / loading JacobLinCool/VoiceBank-DEMAND-16k ...")
    hf = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

    train_ds = VoiceBankDemandDataset(hf["train"], clip_len=clip_len, window_len=window_len,
                                      mode=mode, pad_short=pad_short)
    val_ds   = VoiceBankDemandDataset(hf["test"],  clip_len=clip_len, window_len=window_len,
                                      mode=mode, pad_short=pad_short)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader

def load_voicebank_demand(cache_dir=None):
    """
        Copyright (c) 2023 Yexin Lu (MP-SENet)
        Copyright (c) 2025-2026 Clément Laroche
        function from : https://github.com/LarocheC/eco8-neaixt
        Load JacobLinCool/VoiceBank-DEMAND-16k (train + test splits).
    """
    from datasets import load_dataset
    return load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", cache_dir=cache_dir)

def collate_fn_pad(batch):
    cleans, noisys = zip(*batch)
    max_len = max(x.shape[-1] for x in cleans)
    cleans = torch.stack([F.pad(x, (0, max_len - x.shape[-1])) for x in cleans])
    noisys = torch.stack([F.pad(x, (0, max_len - x.shape[-1])) for x in noisys])
    return cleans, noisys

def make_vbd_loaders(
    batch_size:  int  = 16,
    clip_len:    int  = CLIP_LEN,
    num_workers: int  = 4,
    pin_memory:  bool = True,
):
    from torch.utils.data import DataLoader
    print("Downloading / loading JacobLinCool/VoiceBank-DEMAND-16k ...")
    hf = load_voicebank_demand()
    print("Creating training dataset ...")
    trainset = VBDDataset(hf['train'], clip_len,
                       split=True, shuffle=True)
    train_loader = DataLoader(trainset, shuffle=False,
                              num_workers=num_workers,
                              batch_size=batch_size,
                              pin_memory=pin_memory,
                              drop_last=True)
    
    print("Creating validation dataset ...")
    validset = VBDDataset(hf['test'], clip_len, split=False, shuffle=False)
    val_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       batch_size=batch_size//2,
                                       pin_memory=True,
                                       drop_last=True,
                                       collate_fn=collate_fn_pad)
    return train_loader, val_loader

# ---------------------------------------------------------------------------
# Quick smoke-test  (run: python voicebank_dataset.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    print("Loading dataset ...")
    hf = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

    for mode in ("chunk", "crop"):
        for pad_short in (True, False):
            ds = VoiceBankDemandDataset(
                hf["train"], clip_len=32_000,
                mode=mode, pad_short=pad_short,
            )
            loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
            noisy, clean = next(iter(loader))
            print(
                f"  mode={mode!r:6s} pad_short={pad_short!s:5s} | "
                f"noisy={tuple(noisy.shape)} dtype={noisy.dtype} | "
                f"clean={tuple(clean.shape)} dtype={clean.dtype}"
            )