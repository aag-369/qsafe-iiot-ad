"""Trains the attack-*type* GRU classifier: benign vs. eavesdrop vs. jamming
vs. pns (see qkd_sim/qber_stream_multiclass.py for what each type models).

This is a second, separate model from the core binary intrusion detector
(ai_detector/train.py) and does not replace or retrain it. The switch
controller's escalate/de-escalate decision keeps using the original binary
detector's confidence score, completely unchanged. This classifier is
additive: it tags *what kind* of attack is in progress, used for reporting
and for the fleet correlator (crypto_agility escalation logic never
consults it).

Usage:
    python -m ai_detector.train_attack_type \
        --train data/qber_multiclass_train.csv --test data/qber_multiclass_test.csv \
        --out models/attack_type_gru.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from ai_detector.features import WindowConfig, build_windows, normalize_features
from ai_detector.model import build_gru_model, count_params
from qkd_sim.qber_stream_multiclass import ATTACK_TYPE_NAMES, AttackType

CLASS_NAMES = [ATTACK_TYPE_NAMES[AttackType(i)] for i in range(4)]


def load_stream(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/qber_multiclass_train.csv")
    parser.add_argument("--test", default="data/qber_multiclass_test.csv")
    parser.add_argument("--out", default="models/attack_type_gru.keras")
    parser.add_argument("--metrics-out", default="models/attack_type_metrics.json")
    parser.add_argument("--norm-out", default="models/attack_type_norm_stats.json")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # This classifier gets one extra feature vs. the binary detector:
    # qber_rolling_mean. Distinguishing attack *types* turns out to hinge
    # on the window's average QBER level (eavesdrop/jamming/pns occupy
    # well-separated mean bands) much more than on delta/variance alone —
    # giving the model that average directly, rather than making it infer
    # one from 20 raw samples, is what closes most of the gap (see git
    # history: macro F1 went from ~0.53 with per-round-jitter data, to
    # ~0.66 after fixing windows to have one coherent per-episode
    # intensity, to notably higher once this feature was added).
    win_cfg = WindowConfig(
        window_size=args.window_size,
        features=("qber", "qber_delta", "qber_rolling_std", "qber_rolling_mean"),
        rolling_window=10,
    )

    train_df = load_stream(args.train)
    test_df = load_stream(args.test)

    X_all, y_all = build_windows(train_df, win_cfg, label_col="attack_type")
    X_test, y_test = build_windows(test_df, win_cfg, label_col="attack_type")

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
    )

    X_train, mean, std = normalize_features(X_train)
    X_val, _, _ = normalize_features(X_val, mean=mean, std=std)
    X_test, _, _ = normalize_features(X_test, mean=mean, std=std)

    n_features = X_train.shape[-1]
    n_classes = len(CLASS_NAMES)
    model = build_gru_model(window_size=args.window_size, n_features=n_features, n_classes=n_classes)
    print(model.summary())
    print("Trainable params:", count_params(model))

    # Inverse-frequency class weights — benign rounds vastly outnumber any
    # single attack type, same imbalance issue as the binary detector.
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    total = len(y_train)
    class_weight = {
        i: (total / (n_classes * counts[i])) if counts[i] > 0 else 1.0
        for i in range(n_classes)
    }
    print("Class weights:", class_weight)

    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=6, restore_best_weights=True
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # --- Evaluation on held-out test stream ---
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes))).tolist()

    print(f"\nTest set — Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)

    metrics = {
        "class_names": CLASS_NAMES,
        "macro_f1": macro_f1,
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in CLASS_NAMES
        },
        "accuracy": report["accuracy"],
        "confusion_matrix": cm,
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
