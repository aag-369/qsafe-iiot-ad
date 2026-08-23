"""Trains the GRU QBER-anomaly detector on the generated telemetry stream
and evaluates it (F1, precision, recall) on a held-out stream.

Usage:
    python -m ai_detector.train \
        --train data/qber_train.csv --test data/qber_test.csv \
        --out models/gru_detector.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ai_detector.features import WindowConfig, build_windows, normalize_features
from ai_detector.model import build_gru_model, count_params


def load_stream(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/qber_train.csv")
    parser.add_argument("--test", default="data/qber_test.csv")
    parser.add_argument("--out", default="models/gru_detector.keras")
    parser.add_argument("--metrics-out", default="models/train_metrics.json")
    parser.add_argument("--norm-out", default="models/norm_stats.json")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(args.seed)

    win_cfg = WindowConfig(window_size=args.window_size)

    train_df = load_stream(args.train)
    test_df = load_stream(args.test)

    X_all, y_all = build_windows(train_df, win_cfg)
    X_test, y_test = build_windows(test_df, win_cfg)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
    )

    X_train, mean, std = normalize_features(X_train)
    X_val, _, _ = normalize_features(X_val, mean=mean, std=std)
    X_test, _, _ = normalize_features(X_test, mean=mean, std=std)

    n_features = X_train.shape[-1]
    model = build_gru_model(window_size=args.window_size, n_features=n_features)
    print(model.summary())
    print("Trainable params:", count_params(model))

    # Handle class imbalance (attack windows are the minority class).
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    class_weight = {
        0: (len(y_train) / (2.0 * n_neg)) if n_neg > 0 else 1.0,
        1: (len(y_train) / (2.0 * n_pos)) if n_pos > 0 else 1.0,
    }
    print("Class weights:", class_weight)

    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6, restore_best_weights=True
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # --- Calibrate decision threshold on the validation split ---
    # The paper's confidence-gated escalation needs a single operating
    # threshold; we pick the one maximizing F1 on validation data (never on
    # test) and apply it, frozen, to the held-out test stream.
    y_val_prob = model.predict(X_val, verbose=0).flatten()
    best_threshold, best_val_f1 = args.threshold, -1.0
    for thr in np.arange(0.05, 0.96, 0.01):
        f1_thr = f1_score(y_val, (y_val_prob >= thr).astype(int), zero_division=0)
        if f1_thr > best_val_f1:
            best_val_f1 = f1_thr
            best_threshold = float(thr)
    print(f"Calibrated threshold (max F1 on val): {best_threshold:.2f} (val F1={best_val_f1:.4f})")
    args.threshold = best_threshold

    # --- Evaluation on held-out test stream ---
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= args.threshold).astype(int)

    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\nTest set — F1: {f1:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  AUC: {auc:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)

    metrics = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc,
        "threshold": args.threshold,
        "n_train_windows": int(len(X_train)),
        "n_val_windows": int(len(X_val)),
        "n_test_windows": int(len(X_test)),
        "trainable_params": count_params(model),
        "window_size": args.window_size,
        "features": list(win_cfg.features),
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    norm_stats = {"mean": mean.tolist(), "std": std.tolist(), "features": list(win_cfg.features)}
    with open(args.norm_out, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"Saved model -> {args.out}")
    print(f"Saved metrics -> {args.metrics_out}")
    print(f"Saved normalization stats -> {args.norm_out}")


if __name__ == "__main__":
    main()
