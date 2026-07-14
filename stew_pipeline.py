"""
stew_pipeline.py

Download the STEW (Simultaneous Task EEG Workload) dataset from Kaggle,
segment each subject's resting ("lo") and multitasking ("hi") recordings
into fixed-length epochs, extract per-channel band-power features, and
save the resulting (X, y, groups) arrays to a compressed .npz file.

Dataset
-------
Kaggle: mitulahirwal/mental-cognitive-workload-eeg-data-stew-dataset
48 subjects, each with two files:
    subNN_lo.txt  -> ~2.5 min resting-state EEG
    subNN_hi.txt  -> ~2.5 min EEG during the SIMKAP multitasking test
Recorded with a 14-channel Emotiv EPOC headset at 128 Hz. Each row is one
sample; each column is one channel, in this order:
    AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4

Usage
-----
    python stew_pipeline.py

Configuration is done via the constants below, or environment variables:
    STEW_DATA_DIR  -> where raw .txt files live / will be downloaded to
    STEW_OUT_PATH  -> where to write the features .npz

Requirements
------------
    pip install numpy scipy kaggle

Kaggle auth (only needed for auto-download)
--------------------------------------------
    Requires ~/.kaggle/kaggle.json (API token), or the KAGGLE_USERNAME /
    KAGGLE_KEY environment variables. See https://www.kaggle.com/docs/api

If you already have the dataset on disk, just point DATA_DIR at the folder
containing the subNN_lo/hi.txt files and download_stew() will be skipped
automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.signal import welch

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

KAGGLE_DATASET = "mitulahirwal/mental-cognitive-workload-eeg-data-stew-dataset"

# Where the raw STEW .txt files live (or will be downloaded to).
DATA_DIR = Path(os.environ.get("STEW_DATA_DIR", "data/stew"))

# Where to write the extracted feature/label arrays.
OUT_PATH = Path(os.environ.get("STEW_OUT_PATH", "data/stew_features.npz"))

CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

FS = 128  # Hz, Emotiv EPOC sampling rate used in STEW

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

EPOCH_SEC = 2.0       # length of each analysis window, in seconds
EPOCH_OVERLAP = 0.5   # fraction of overlap between consecutive windows
LABELS = {"lo": 0, "hi": 1}  # lo = resting, hi = SIMKAP multitasking workload


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download_stew(data_dir: Path = DATA_DIR) -> None:
    """Download + unzip the STEW dataset from Kaggle into data_dir, if needed."""
    data_dir = Path(data_dir)
    existing = list(data_dir.glob("sub*_*.txt")) if data_dir.exists() else []
    if existing:
        print(f"Found {len(existing)} existing STEW files in {data_dir}, skipping download.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The 'kaggle' package is required to auto-download the dataset. "
            "Install it with `pip install kaggle`, put your API token at "
            "~/.kaggle/kaggle.json, and rerun -- or manually download the "
            f"dataset from https://www.kaggle.com/datasets/{KAGGLE_DATASET} "
            f"and place the .txt files in {data_dir}."
        ) from exc

    print(f"Downloading {KAGGLE_DATASET} to {data_dir} ...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(data_dir), unzip=True, quiet=False)

    # Kaggle sometimes nests files one directory deeper; flatten if so.
    for nested_path in data_dir.rglob("sub*_*.txt"):
        target = data_dir / nested_path.name
        if nested_path != target:
            nested_path.rename(target)

    print("Download complete.")


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def _band_power(epoch: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    epoch: (n_channels, n_samples) array for one epoch.
    Returns a flattened (n_channels * n_bands,) array of log band powers.
    """
    nperseg = min(fs * 2, epoch.shape[-1])
    freqs, psd = welch(epoch, fs=fs, nperseg=nperseg, axis=-1)
    feats = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        band_power = psd[:, mask].mean(axis=-1)
        feats.append(band_power)
    feats = np.stack(feats, axis=-1)  # (n_channels, n_bands)
    return np.log1p(feats).reshape(-1)  # -> (n_channels * n_bands,)


def _epoch_signal(signal: np.ndarray, fs: int = FS,
                   epoch_sec: float = EPOCH_SEC, overlap: float = EPOCH_OVERLAP):
    """
    signal: (n_samples, n_channels)
    Yields (n_channels, epoch_len) windows.
    """
    epoch_len = int(epoch_sec * fs)
    step = max(1, int(epoch_len * (1 - overlap)))
    n_samples = signal.shape[0]
    for start in range(0, n_samples - epoch_len + 1, step):
        yield signal[start:start + epoch_len].T  # (n_channels, epoch_len)


def _load_subject_file(path: Path) -> np.ndarray:
    """Load a single subNN_lo/hi.txt file -> (n_samples, n_channels) array."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != len(CHANNELS):
        raise ValueError(
            f"{path} has shape {data.shape}, expected (n_samples, {len(CHANNELS)}) "
            f"for channels {CHANNELS}"
        )
    return data


# --------------------------------------------------------------------------- #
# Build dataset
# --------------------------------------------------------------------------- #

def build_dataset(data_dir: Path = DATA_DIR):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("sub*_*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No STEW .txt files found in {data_dir}. Call download_stew() first, "
            f"or point DATA_DIR at an existing copy of the dataset."
        )

    X, y, groups = [], [], []

    for path in files:
        stem = path.stem  # e.g. "sub01_lo"
        parts = stem.split("_")
        if len(parts) != 2 or parts[1].lower() not in LABELS:
            print(f"Skipping unrecognized file name: {path.name}")
            continue

        sub_str, task = parts
        digits = "".join(ch for ch in sub_str if ch.isdigit())
        if not digits:
            print(f"Skipping file with no subject id: {path.name}")
            continue
        subject_id = int(digits)
        label = LABELS[task.lower()]

        signal = _load_subject_file(path)

        n_epochs = 0
        for epoch in _epoch_signal(signal):
            X.append(_band_power(epoch))
            y.append(label)
            groups.append(subject_id)
            n_epochs += 1

        print(f"{path.name}: subject={subject_id} label={task} epochs={n_epochs}")

    if not X:
        raise RuntimeError(f"No epochs were extracted from files in {data_dir}.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)

    return X, y, groups


def main():
    download_stew(DATA_DIR)
    X, y, groups = build_dataset(DATA_DIR)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        X=X,
        y=y,
        groups=groups,
        channels=np.array(CHANNELS),
        bands=np.array(list(BANDS.keys())),
    )

    print(f"Saved {X.shape[0]} epochs x {X.shape[1]} features to {OUT_PATH}")
    counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"Subjects: {len(np.unique(groups))}, class balance: {counts}")


if __name__ == "__main__":
    main()
