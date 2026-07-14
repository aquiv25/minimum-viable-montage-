"""
stew_pipeline.py

Load the STEW (Simultaneous Task EEG Workload) dataset -- as distributed on
Kaggle in pre-packaged .mat form -- segment each subject's SIMKAP-multitasking
recording into fixed-length epochs, extract per-channel band-power features,
and save the resulting (X, y, groups) arrays to a compressed .npz file.

Dataset
-------
Kaggle: mitulahirwal/mental-cognitive-workload-eeg-data-stew-dataset

This mirror packages the original 48-subject STEW recordings (Emotiv EPOC,
14 channels, 128 Hz, 2.5 min SIMKAP multitasking task) as four .mat files,
after dropping 3 subjects (5, 24, 42) whose subjective ratings are missing:

    dataset.mat              -> "dataset", shape (14, 19200, 45)
                                 14 channels x 19200 samples x 45 subjects
    rating.mat                -> "rating", shape (45, 1)
                                 subjective workload rating, 1-9 scale
    class_012.mat              -> "class_012", shape (45, 1)
                                 3-class label derived from rating:
                                   0 = normal   (rating 4-5)
                                   1 = moderate (rating 6-7)
                                   2 = high     (rating 8-9)
    three_class_one_hot.mat -> "three_class_one_hot", shape (45, 3)
                                 one-hot encoding of class_012 (unused here)

IMPORTANT: this mirror only ships ONE recording per subject -- the SIMKAP
task recording -- not a paired rest recording, so there's no rest-vs-task
axis available here at all, only workload-intensity within the task.
That matches the project's actual target (overload detection during a task),
not a rest/task split.

Also note: class_012's bins (4-5 / 6-7 / 8-9) do NOT match Lim et al.'s
published low/moderate/high scheme (1-3 / 4-6 / 7-9) -- this dataset mirror
simply has no subjects rating 1-3 during the task. For comparability with
Lim et al. and with the project's "overload" framing (trigger a break at
high load), prefer "two_class" (rating >= 7 -> overloaded), which reuses
Lim et al.'s own hi-band cutoff, over the pre-baked "three_class" labels.

Channel order is assumed to match the original STEW recordings:
    AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4

Usage
-----
    python stew_pipeline.py

Configuration is done via the constants below, or environment variables:
    STEW_DATA_DIR    -> where the .mat files live / will be downloaded to
    STEW_OUT_PATH    -> where to write the features .npz
    STEW_LABEL_MODE  -> "three_class" (default) or "two_class"
                        two_class: 0 = normal (rating <= 6), 1 = high (rating >= 7)

Requirements
------------
    pip install numpy scipy scikit-learn kaggle

Kaggle auth (only needed for auto-download)
--------------------------------------------
    Requires ~/.kaggle/kaggle.json (API token), or the KAGGLE_USERNAME /
    KAGGLE_KEY environment variables. See https://www.kaggle.com/docs/api

If you already have the dataset on disk, just point DATA_DIR at the folder
containing dataset.mat / rating.mat / class_012.mat and download_stew() will
be skipped automatically.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

KAGGLE_DATASET = "mitulahirwal/mental-cognitive-workload-eeg-data-stew-dataset"

# Where the .mat files live (or will be downloaded to).
DATA_DIR = Path(os.environ.get("STEW_DATA_DIR", "data/stew"))

# Where to write the extracted feature/label arrays.
OUT_PATH = Path(os.environ.get("STEW_OUT_PATH", "data/stew_features.npz"))

# "three_class" (0/1/2, per class_012.mat) or "two_class" (0/1, re-derived from rating).
# two_class is the project's primary target -- see note above on why.
LABEL_MODE = os.environ.get("STEW_LABEL_MODE", "two_class")

REQUIRED_FILES = ["dataset.mat", "rating.mat", "class_012.mat"]

CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

# Channels realistic for a frontal/temporal wearable headband (excludes parietal/
# occipital: P7, P8, O1, O2). This is the pool the ablation search is restricted to.
FRONTAL_TEMPORAL = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6", "T7", "T8"]

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


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download_stew(data_dir: Path = DATA_DIR) -> None:
    """Download + unzip the STEW dataset from Kaggle into data_dir, if needed."""
    data_dir = Path(data_dir)
    if data_dir.exists() and all((data_dir / f).exists() for f in REQUIRED_FILES):
        print(f"Found existing STEW .mat files in {data_dir}, skipping download.")
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
            f"and place {REQUIRED_FILES} in {data_dir}."
        ) from exc

    print(f"Downloading {KAGGLE_DATASET} to {data_dir} ...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(data_dir), unzip=True, quiet=False)

    # Kaggle sometimes nests files one directory deeper; flatten if so.
    for name in REQUIRED_FILES:
        nested = next(data_dir.rglob(name), None)
        if nested and nested != data_dir / name:
            nested.rename(data_dir / name)
    print("Download complete.")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_raw(data_dir: Path = DATA_DIR):
    """
    Returns:
        signals: (n_subjects, n_channels, n_samples) float array
        rating:  (n_subjects,) int array, 1-9 subjective workload rating
        class3:  (n_subjects,) int array, 0/1/2 three-class label from class_012.mat
    """
    data_dir = Path(data_dir)
    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {data_dir}. Call download_stew() first, or "
            f"point DATA_DIR at an existing copy of the dataset."
        )

    dataset = loadmat(data_dir / "dataset.mat")["dataset"]  # (14, 19200, 45)
    signals = np.transpose(dataset, (2, 0, 1))              # (45, 14, 19200)
    rating = loadmat(data_dir / "rating.mat")["rating"].ravel().astype(np.int64)
    class3 = loadmat(data_dir / "class_012.mat")["class_012"].ravel().astype(np.int64)

    if signals.shape[1] != len(CHANNELS):
        raise ValueError(
            f"dataset.mat has {signals.shape[1]} channels, expected {len(CHANNELS)} "
            f"({CHANNELS})"
        )
    return signals, rating, class3


def compute_labels(rating: np.ndarray, class3: np.ndarray, mode: str = LABEL_MODE) -> np.ndarray:
    if mode == "three_class":
        return class3
    if mode == "two_class":
        # Per Lim et al.'s original hi-band cutoff: high/overloaded = rating 7-9.
        return (rating >= 7).astype(np.int64)
    raise ValueError(f"Unknown STEW_LABEL_MODE: {mode!r} (expected 'three_class' or 'two_class')")


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def _band_power(epoch: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    epoch: (n_channels, n_samples) array for one epoch.
    Returns a flattened (n_channels * n_bands,) array of log band powers,
    ordered [ch0_band0, ch0_band1, ..., ch1_band0, ...] (channel-major).
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
    signal: (n_channels, n_samples)
    Yields (n_channels, epoch_len) windows.
    """
    epoch_len = int(epoch_sec * fs)
    step = max(1, int(epoch_len * (1 - overlap)))
    n_samples = signal.shape[-1]
    for start in range(0, n_samples - epoch_len + 1, step):
        yield signal[:, start:start + epoch_len]


def feature_names(channels=CHANNELS, bands=BANDS) -> list:
    """Flat feature name list matching _band_power's channel-major ordering."""
    return [f"{ch}_{band}" for ch in channels for band in bands]


# --------------------------------------------------------------------------- #
# Build dataset
# --------------------------------------------------------------------------- #

def build_dataset(data_dir: Path = DATA_DIR, label_mode: str = LABEL_MODE):
    signals, rating, class3 = load_raw(data_dir)
    labels = compute_labels(rating, class3, mode=label_mode)

    X, y, groups = [], [], []
    for subject_idx in range(signals.shape[0]):
        n_epochs = 0
        for epoch in _epoch_signal(signals[subject_idx]):
            X.append(_band_power(epoch))
            y.append(labels[subject_idx])
            groups.append(subject_idx)
            n_epochs += 1
        print(f"subject={subject_idx} rating={rating[subject_idx]} "
              f"label={labels[subject_idx]} epochs={n_epochs}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    return X, y, groups


def _save(out_path: Path, X, y, groups, label_mode: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        groups=groups,
        channels=np.array(CHANNELS),
        bands=np.array(list(BANDS.keys())),
        feature_names=np.array(feature_names()),
        label_mode=np.array(label_mode),
    )
    counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"Saved {X.shape[0]} epochs x {X.shape[1]} features to {out_path}")
    print(f"Subjects: {len(np.unique(groups))}, label_mode={label_mode}, class balance: {counts}")


def main():
    download_stew(DATA_DIR)

    # Primary target: two_class (overloaded vs not, rating >= 7). This is what
    # stew_baseline.py / stew_loso_ablation.py load by default.
    X2, y2, groups2 = build_dataset(DATA_DIR, "two_class")
    _save(DATA_DIR / "stew_features_two_class.npz", X2, y2, groups2, "two_class")

    # Secondary/exploratory: three_class, using the Kaggle mirror's own bins
    # (4-5/6-7/8-9 -- note this differs from Lim et al.'s published 1-3/4-6/7-9).
    X3, y3, groups3 = build_dataset(DATA_DIR, "three_class")
    _save(DATA_DIR / "stew_features_three_class.npz", X3, y3, groups3, "three_class")

    # Also honor STEW_OUT_PATH / STEW_LABEL_MODE if explicitly set, for one-off runs.
    if "STEW_OUT_PATH" in os.environ or "STEW_LABEL_MODE" in os.environ:
        Xc, yc, groupsc = build_dataset(DATA_DIR, LABEL_MODE)
        _save(OUT_PATH, Xc, yc, groupsc, LABEL_MODE)


if __name__ == "__main__":
    main()
