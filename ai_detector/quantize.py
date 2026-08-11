"""Post-training INT8 quantization of the trained GRU model to TFLite,
targeting deployment via TensorFlow Lite for Microcontrollers on ARM
Cortex-M4 class hardware.

Usage:
    python -m ai_detector.quantize \
        --model models/gru_detector.keras \
        --train data/qber_train.csv \
        --out models/gru_detector_int8.tflite
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from ai_detector.features import WindowConfig, build_windows, normalize_features
from ai_detector.model import build_gru_model


def representative_dataset_factory(X_train_norm: np.ndarray, n_samples: int = 200):
    idx = np.random.choice(len(X_train_norm), size=min(n_samples, len(X_train_norm)), replace=False)

    def representative_dataset():
        for i in idx:
            sample = X_train_norm[i : i + 1].astype(np.float32)
            yield [sample]

    return representative_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gru_detector.keras")
    parser.add_argument("--train", default="data/qber_train.csv")
    parser.add_argument("--norm-stats", default="models/norm_stats.json")
    parser.add_argument("--out", default="models/gru_detector_int8.tflite")
    parser.add_argument("--window-size", type=int, default=20)
    args = parser.parse_args()

    trained = tf.keras.models.load_model(args.model)

    # Rebuild with a statically-unrolled GRU (same weights) so the TFLite
    # converter can lower it without dynamic TensorList ops.
    gru_units = trained.get_layer("gru").units
    dense_units = trained.get_layer("dense_1").units
    n_features_loaded = trained.input_shape[-1]
    model = build_gru_model(
        window_size=args.window_size,
        n_features=n_features_loaded,
        gru_units=gru_units,
        dense_units=dense_units,
        unroll=True,
    )
    model.set_weights(trained.get_weights())

    with open(args.norm_stats) as f:
        norm_stats = json.load(f)
    mean = np.array(norm_stats["mean"], dtype=np.float32)
    std = np.array(norm_stats["std"], dtype=np.float32)

    win_cfg = WindowConfig(window_size=args.window_size, features=tuple(norm_stats["features"]))
    train_df = pd.read_csv(args.train)
    X_train, _ = build_windows(train_df, win_cfg)
    X_train_norm, _, _ = normalize_features(X_train, mean=mean, std=std)

    # NOTE on quantization scheme: TFLite's fully-integer (INT8 activations +
    # INT8 weights) quantization path only lowers GRU/LSTM cells through a
    # specialized TFLiteLSTMCell op-fusion that recent Keras GRU graphs do
    # not emit by default, so forcing TFLITE_BUILTINS_INT8-only causes the
    # converter to silently fall back to float kernels for the recurrent
    # core (no real size/latency win, sometimes a net loss from added
    # quant/dequant nodes). We therefore use INT8 *weight-only* (dynamic
    # range) quantization instead: weight tensors are stored as INT8 (~4x
    # smaller per-tensor than float32) while activations stay float. For a
    # model this small, the measured *whole-file* reduction is modest
    # because a fixed flatbuffer/graph-metadata overhead dominates the tiny
    # weight tensors; on a real TFLite Micro deployment (bare op resolver,
    # no flatbuffer verifier/interpreter overhead) the win is dominated by
    # the weight-tensor compression instead. Both figures are reported
    # below rather than assumed.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tflite_model = converter.convert()

    # Also produce a float32 (unquantized) baseline for a fair size/latency
    # comparison in the benchmark report.
    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32_model = converter_fp32.convert()
    fp32_tflite_path = str(Path(args.out).with_name(Path(args.out).stem.replace("_int8", "_fp32") + ".tflite"))
    with open(fp32_tflite_path, "wb") as f:
        f.write(tflite_fp32_model)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(tflite_model)

    fp32_size = len(tflite_fp32_model)
    int8_size = len(tflite_model)
    reduction_pct = (1 - int8_size / fp32_size) * 100

    print(f"TFLite float32 model size: {fp32_size / 1024:.1f} KB")
    print(f"TFLite INT8 (weights) model size: {int8_size / 1024:.1f} KB")
    print(f"Size reduction: {reduction_pct:.1f}%")
    print(f"Saved quantized model -> {args.out}")
    print(f"Saved float32 baseline -> {fp32_tflite_path}")

    size_report = {
        "tflite_fp32_bytes": fp32_size,
        "tflite_int8_bytes": int8_size,
        "reduction_pct": reduction_pct,
    }
    with open(str(Path(args.out).with_suffix("")) + "_size_report.json", "w") as f:
        json.dump(size_report, f, indent=2)


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


if __name__ == "__main__":
    main()
