"""Runs the trained attack-*type* GRU classifier over a full QBER stream,
producing one predicted attack-type label (and its class probabilities) per
round — the additive companion to detector_runner.py's binary confidence
score. Never consulted by the switch controller; used only for tagging and
fleet-level correlation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import tensorflow as tf

from ai_detector.features import WindowConfig, build_windows, normalize_features
from qkd_sim.qber_stream_multiclass import AttackType

_TYPE_WINDOW_FEATURES = ("qber", "qber_delta", "qber_rolling_std", "qber_rolling_mean")


class AttackTypeRunner:
    def __init__(
        self,
        model_path: str = "models/attack_type_gru.keras",
        norm_stats_path: str = "models/attack_type_norm_stats.json",
        window_size: int = 20,
    ):
        self.model = tf.keras.models.load_model(model_path)
        with open(norm_stats_path) as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float32)
        self.std = np.array(stats["std"], dtype=np.float32)
        self.win_cfg = WindowConfig(
            window_size=window_size,
            features=tuple(stats["features"]),
            rolling_window=10,
        )

    def score_stream(self, df: pd.DataFrame) -> np.ndarray:
        """Returns an (len(df), 4) array of class probabilities
        [benign, eavesdrop, jamming, pns] per round. Rounds before the first
        full window default to [1, 0, 0, 0] (benign), matching how a real
        deployment behaves while its telemetry buffer fills up — identical
        convention to DetectorRunner.score_stream's zero-confidence default."""
        # attack_type column may not exist on live-generated frames scored
        # purely for inference; build_windows only reads it for the (here
        # unused) label array, so a dummy column is enough.
        df = df.copy()
        if "attack_type" not in df.columns:
            df["attack_type"] = 0
        X, _ = build_windows(df, self.win_cfg, label_col="attack_type")
        X_norm, _, _ = normalize_features(X, mean=self.mean, std=self.std)
        probs = self.model.predict(X_norm, verbose=0)

        n_classes = probs.shape[-1]
        out = np.zeros((len(df), n_classes), dtype=np.float32)
        out[:, AttackType.BENIGN] = 1.0
        start = self.win_cfg.window_size - 1
        out[start : start + len(probs)] = probs
        return out

    def predict_types(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the argmax attack-type index per round (int array)."""
        return np.argmax(self.score_stream(df), axis=1)
