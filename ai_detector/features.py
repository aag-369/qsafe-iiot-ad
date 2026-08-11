"""Turns a raw QBER time series (as produced by qkd_sim) into fixed-length
sliding windows suitable for GRU sequence classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowConfig:
    window_size: int = 20
    stride: int = 1
    # Feature set kept small to minimize the input tensor footprint on a
    # Cortex-M4 target, while giving the GRU enough signal to separate
    # stealthy (low intercept-rate) attacks from benign channel jitter.
    features: tuple[str, ...] = ("qber", "qber_delta", "qber_rolling_std")
    rolling_window: int = 5


def _engineer_features(df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    df = df.copy()
    df["qber_delta"] = df["qber"].diff().fillna(0.0)
    # Causal rolling std (only past values) — a stealthy eavesdropper raises
    # local variance even when the mean QBER stays under a naive threshold.
    df["qber_rolling_std"] = (
        df["qber"].rolling(window=rolling_window, min_periods=1).std().fillna(0.0)
    )
    return df


def build_windows(
    df: pd.DataFrame,
    config: WindowConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds (X, y) arrays of shape (n_windows, window_size, n_features) and
    (n_windows,). The label for each window is the ground-truth attack label
    at the *final* timestep of the window (i.e. "is an intrusion in progress
    right now, given the recent QBER history").
    """
    config = config or WindowConfig()
    df = _engineer_features(df, config.rolling_window)

    feat_matrix = df[list(config.features)].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.float32)

    n = len(df)
    windows = []
    window_labels = []
    for end in range(config.window_size, n + 1, config.stride):
        start = end - config.window_size
        windows.append(feat_matrix[start:end])
        window_labels.append(labels[end - 1])

    X = np.stack(windows).astype(np.float32)
    y = np.array(window_labels, dtype=np.float32)
    return X, y


def normalize_features(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature standardization. If mean/std are not given, they are
    computed from X (use this on the training set, then reuse the returned
    mean/std on validation/test/deployment data)."""
    if mean is None:
        mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    if std is None:
        std = X.reshape(-1, X.shape[-1]).std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
    X_norm = (X - mean) / std
    return X_norm.astype(np.float32), mean, std
