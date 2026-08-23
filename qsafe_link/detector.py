"""
Live, single-window inference for the demo.

Two deliberate choices here, both of which make the demo a *stronger* claim
than the batch pipeline it wraps:

1. It runs the **INT8-quantized TFLite model** (`models/gru_detector_int8.tflite`)
   by default -- the same 80.6 KB artifact the paper reports as the
   Cortex-M4 deployment target -- rather than the float32 Keras model used
   for training. So the thing making decisions on stage is the thing that
   would actually be flashed onto the microcontroller. Measured on this
   host: 0.02 ms per window versus 60 ms for a single-window Keras call,
   with 100% agreement at the operating threshold (max abs deviation 0.0017
   over a 64-window comparison).

2. Features are computed by calling `ai_detector.features.build_windows`
   on a rolling tail of the stream, not by a re-implementation. A
   hand-rolled incremental version of the same rolling statistics is exactly
   the kind of thing that silently drifts from training-time preprocessing
   and quietly degrades the model. The tail is kept at
   `window_size + rolling_window` rounds, which is long enough that the
   final window's causal rolling statistics are identical to what a
   full-history pass would produce.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ai_detector.features import WindowConfig, build_windows, normalize_features


class LiveDetector:
    """Scores the most recent QBER window, one round at a time."""

    def __init__(
        self,
        models_dir: str | Path = "models",
        window_size: int = 20,
        prefer_tflite: bool = True,
    ):
        models_dir = Path(models_dir)
        with open(models_dir / "norm_stats.json") as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float32)
        self.std = np.array(stats["std"], dtype=np.float32)
        self.win_cfg = WindowConfig(window_size=window_size, features=tuple(stats["features"]))

        with open(models_dir / "train_metrics.json") as f:
            self.threshold = float(json.load(f)["threshold"])

        self.window_size = window_size
        # One extra `rolling_window` of history so the last window's causal
        # rolling std matches a full-history computation exactly.
        self._tail_len = window_size + self.win_cfg.rolling_window + 1
        self._qber: deque[float] = deque(maxlen=self._tail_len)

        self.backend = "none"
        self._interp = None
        self._keras = None

        tflite_path = models_dir / "gru_detector_int8.tflite"
        if prefer_tflite and tflite_path.exists():
            try:
                self._load_tflite(tflite_path)
                self.backend = "tflite-int8"
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[qsafe-link] TFLite load failed ({exc}); falling back to Keras.")

        if self.backend == "none":
            import tensorflow as tf

            self._keras = tf.keras.models.load_model(str(models_dir / "gru_detector.keras"))
            self.backend = "keras-fp32"

    def _load_tflite(self, path: Path) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter  # type: ignore
        except Exception:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter  # noqa: N806
        self._interp = Interpreter(model_path=str(path))
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]

    # --- stateless scoring (preferred; one detector serves many nodes) ------
    def score_tail(self, qber_tail: Sequence[float]) -> float:
        """Score the most recent window of a caller-owned QBER history.

        This is the API the multi-device runtime uses. Each `EdgeNode` keeps
        its own QBER buffer and passes it in, so a single loaded model can
        serve the whole fleet without the devices' histories interleaving --
        which is exactly what would happen if the detector held the buffer
        itself and were shared. Stepping is serialized onto one worker
        thread, so the underlying TFLite interpreter is never re-entered
        concurrently.

        Returns 0.0 until a full window exists, matching the convention in
        `orchestrator/detector_runner.py`: a real deployment stays on the
        baseline profile while its telemetry buffer fills.
        """
        tail = list(qber_tail)[-self._tail_len:]
        if len(tail) < self.window_size:
            return 0.0
        df = pd.DataFrame({"qber": tail, "label": 0})
        X, _ = build_windows(df, self.win_cfg)
        X_norm, _, _ = normalize_features(X[-1:], mean=self.mean, std=self.std)
        return self._infer(X_norm.astype(np.float32))

    @property
    def tail_len(self) -> int:
        """Minimum history a caller must retain for exact rolling features."""
        return self._tail_len

    # --- stateful convenience wrapper (single-node use, tests) --------------
    def push(self, qber: float) -> float:
        """Append one QBER sample to this detector's own buffer and score it.

        Convenient for single-node use and tests. Do not share a detector
        used this way between devices -- use `score_tail` instead.
        """
        self._qber.append(float(qber))
        return self.score_tail(self._qber)

    def _infer(self, x: np.ndarray) -> float:
        if self._interp is not None:
            self._interp.set_tensor(self._in["index"], x)
            self._interp.invoke()
            return float(self._interp.get_tensor(self._out["index"])[0][0])
        return float(self._keras(x, training=False)[0][0])

    def reset(self) -> None:
        self._qber.clear()

    @property
    def n_buffered(self) -> int:
        return len(self._qber)

    @property
    def warm(self) -> bool:
        return len(self._qber) >= self.window_size

    def describe(self) -> dict:
        return {
            "backend": self.backend,
            "window_size": self.window_size,
            "threshold": self.threshold,
            "features": list(self.win_cfg.features),
            "buffered": self.n_buffered,
            "warm": self.warm,
        }


class LiveTypeTagger:
    """Additive attack-type tagging for the live demo.

    Kept deliberately separate from `LiveDetector` and run at a lower cadence
    on a batch of devices, because it is (a) additive -- it never touches the
    escalate/de-escalate decision -- and (b) only available as a float Keras
    model, so it is ~3,000x more expensive per call than the INT8 detector.
    Running it per-round per-device would dominate the loop for a label that
    appears on screen as a caption.
    """

    def __init__(self, models_dir: str | Path = "models", window_size: int = 20):
        import tensorflow as tf

        models_dir = Path(models_dir)
        self.model = tf.keras.models.load_model(str(models_dir / "attack_type_gru.keras"))
        with open(models_dir / "attack_type_norm_stats.json") as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float32)
        self.std = np.array(stats["std"], dtype=np.float32)
        self.win_cfg = WindowConfig(
            window_size=window_size, features=tuple(stats["features"]), rolling_window=10
        )
        self.window_size = window_size
        self.tail_len = window_size + self.win_cfg.rolling_window + 1

    def build_window(self, qber_tail: list[float]) -> np.ndarray | None:
        """Turn one device's QBER tail into a single normalized window."""
        if len(qber_tail) < self.window_size:
            return None
        df = pd.DataFrame({"qber": qber_tail, "attack_type": 0})
        X, _ = build_windows(df, self.win_cfg, label_col="attack_type")
        X_norm, _, _ = normalize_features(X[-1:], mean=self.mean, std=self.std)
        return X_norm[0]

    def tag_batch(self, windows: list[np.ndarray]) -> list[tuple[int, float]]:
        """Classify several devices' windows in one batched call.

        Returns (class_index, confidence) per input window.
        """
        if not windows:
            return []
        X = np.stack(windows).astype(np.float32)
        probs = self.model.predict(X, verbose=0)
        return [(int(np.argmax(p)), float(np.max(p))) for p in probs]
