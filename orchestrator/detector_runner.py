"""Runs the trained GRU detector over a full QBER stream, producing one
confidence score per round, aligned 1:1 with the input dataframe rows.

The first `window_size - 1` rounds have no full window of history yet; the
system reports 0.0 confidence (i.e. stays on the baseline profile) for
those rounds, matching what a real deployment would do while its telemetry
buffer fills up.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import tensorflow as tf

from ai_detector.features import WindowConfig, build_windows, normalize_features


class DetectorRunner:
    def __init__(
        self,
        model_path: str = "models/gru_detector.keras",
        norm_stats_path: str = "models/norm_stats.json",
        window_size: int = 20,
    ):
        self.model = tf.keras.models.load_model(model_path)
        with open(norm_stats_path) as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float32)
        self.std = np.array(stats["std"], dtype=np.float32)
        self.win_cfg = WindowConfig(window_size=window_size, features=tuple(stats["features"]))

    def score_stream(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = build_windows(df, self.win_cfg)
        X_norm, _, _ = normalize_features(X, mean=self.mean, std=self.std)
        probs = self.model.predict(X_norm, verbose=0).flatten()

        confidences = np.zeros(len(df), dtype=np.float32)
        # build_windows() emits one row per window ending at index
        # (window_size - 1), (window_size), ..., (len(df) - 1).
        start = self.win_cfg.window_size - 1
        confidences[start : start + len(probs)] = probs
        return confidences
