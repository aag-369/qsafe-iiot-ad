"""Ablation study for the binary QBER-anomaly detector.

Answers two questions the paper needs evidence for:
  1. Does each engineered feature (qber_delta, qber_rolling_std) actually
     earn its place, or would raw QBER alone do just as well?
  2. How much temporal context (window_size) is needed?

Every configuration is trained from scratch with the same seed, the same
class-weighting, and the same validation-set threshold calibration as
ai_detector/train.py, so the numbers are directly comparable to the
deployed model rather than to a differently-tuned baseline.

Usage:
    python -m ai_detector.ablation --out models/ablation_report.json
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

FEATURE_SETS = {
    "qber": ("qber",),
    "qber+delta": ("qber", "qber_delta"),
    "qber+delta+std": ("qber", "qber_delta", "qber_rolling_std"),
}


def run_one(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: tuple[str, ...],
    window_size: int,
    seed: int = 7,
    epochs: int = 30,
    batch_size: int = 64,
) -> dict:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    win_cfg = WindowConfig(window_size=window_size, features=features)

    X_all, y_all = build_windows(train_df, win_cfg)
    X_test, y_test = build_windows(test_df, win_cfg)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
    )
    X_train, mean, std = normalize_features(X_train)
    X_val, _, _ = normalize_features(X_val, mean=mean, std=std)
    X_test, _, _ = normalize_features(X_test, mean=mean, std=std)

    model = build_gru_model(window_size=window_size, n_features=X_train.shape[-1])

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    class_weight = {
        0: (len(y_train) / (2.0 * n_neg)) if n_neg > 0 else 1.0,
        1: (len(y_train) / (2.0 * n_pos)) if n_pos > 0 else 1.0,
    }

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=6, restore_best_weights=True
            )
        ],
        verbose=0,
    )

    # Threshold calibrated on validation only — never on test.
    y_val_prob = model.predict(X_val, verbose=0).flatten()
    best_thr, best_val_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.01):
        f = f1_score(y_val, (y_val_prob >= thr).astype(int), zero_division=0)
        if f > best_val_f1:
            best_val_f1, best_thr = f, float(thr)

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= best_thr).astype(int)

    return {
        "features": list(features),
        "n_features": len(features),
        "window_size": window_size,
        "threshold": best_thr,
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "trainable_params": count_params(model),
    }


def aggregate(runs: list[dict]) -> dict:
    """Mean and sample std across seeds for each metric. Reported because a
    single-seed difference of a few F1 points on this workload is smaller
    than seed-to-seed variance, and quoting one lucky run would overstate
    what the architecture actually buys."""
    keys = ("f1", "precision", "recall", "roc_auc")
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    out["n_seeds"] = len(runs)
    out["trainable_params"] = runs[0]["trainable_params"]
    out["window_size"] = runs[0]["window_size"]
    out["features"] = runs[0]["features"]
    out["per_seed_f1"] = [r["f1"] for r in runs]
    return out


def cmd_runs(args) -> None:
    """Run an explicit list of `featureset:window:seed` specs and append each
    result to a JSONL file. Lets a long sweep be split across several short
    invocations and resumed without losing completed work."""
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    done = set()
    if Path(args.jsonl).exists():
        for line in open(args.jsonl):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add((r["featureset"], r["window_size"], r["seed"]))

    for spec in args.runs.split(","):
        spec = spec.strip()
        if not spec:
            continue
        fs, ws, seed = spec.split(":")
        ws, seed = int(ws), int(seed)
        if (fs, ws, seed) in done:
            print(f"  skip (done): {spec}", flush=True)
            continue
        r = run_one(train_df, test_df, FEATURE_SETS[fs], ws, seed=seed)
        r["featureset"] = fs
        r["seed"] = seed
        with open(args.jsonl, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"  {fs:16s} W={ws:<3d} seed={seed}  F1={r['f1']:.4f} "
              f"AUC={r['roc_auc']:.4f}", flush=True)


def cmd_aggregate(args) -> None:
    """Collapse the JSONL of individual runs into the grouped report the
    paper's tables and figures read."""
    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    results = {"feature_ablation": [], "window_ablation": []}

    for fs in FEATURE_SETS:
        runs = [r for r in rows if r["featureset"] == fs and r["window_size"] == 20]
        if runs:
            agg = aggregate(runs)
            agg["label"] = fs
            results["feature_ablation"].append(agg)

    for ws in (5, 10, 20, 40):
        runs = [r for r in rows
                if r["featureset"] == "qber+delta+std" and r["window_size"] == ws]
        if runs:
            agg = aggregate(runs)
            agg["label"] = f"W={ws}"
            results["window_ablation"].append(agg)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("=== Feature ablation (W=20) ===")
    for r in results["feature_ablation"]:
        print(f"  {r['label']:16s} n={r['n_seeds']} F1={r['f1_mean']:.4f}+-{r['f1_std']:.4f} "
              f"P={r['precision_mean']:.4f} R={r['recall_mean']:.4f} "
              f"AUC={r['roc_auc_mean']:.4f} params={r['trainable_params']}")
    print("=== Window ablation (qber+delta+std) ===")
    for r in results["window_ablation"]:
        print(f"  {r['label']:16s} n={r['n_seeds']} F1={r['f1_mean']:.4f}+-{r['f1_std']:.4f} "
              f"P={r['precision_mean']:.4f} R={r['recall_mean']:.4f} "
              f"AUC={r['roc_auc_mean']:.4f}")
    print(f"\nSaved -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/qber_train.csv")
    ap.add_argument("--test", default="data/qber_test.csv")
    ap.add_argument("--out", default="models/ablation_report.json")
    ap.add_argument("--jsonl", default="models/ablation_runs.jsonl")
    ap.add_argument("--runs", default=None,
                    help="comma-separated featureset:window:seed specs")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--feature-seeds", type=int, default=5)
    ap.add_argument("--window-seeds", type=int, default=3)
    ap.add_argument("--part", choices=["features", "windows", "all"], default="all",
                    help="run one half at a time so each invocation stays short")
    args = ap.parse_args()

    if args.runs:
        return cmd_runs(args)
    if args.aggregate:
        return cmd_aggregate(args)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # Merge into an existing report so the two halves can be run separately.
    results = {"feature_ablation": [], "window_ablation": []}
    if Path(args.out).exists():
        try:
            results.update(json.load(open(args.out)))
        except Exception:
            pass

    if args.part in ("features", "all"):
        print(f"=== Feature ablation (window_size=20, {args.feature_seeds} seeds) ===")
        results["feature_ablation"] = []
        for name, feats in FEATURE_SETS.items():
            runs = [
                run_one(train_df, test_df, feats, 20, seed=7 + s)
                for s in range(args.feature_seeds)
            ]
            agg = aggregate(runs)
            agg["label"] = name
            results["feature_ablation"].append(agg)
            print(f"  {name:18s} F1={agg['f1_mean']:.4f}+-{agg['f1_std']:.4f} "
                  f"P={agg['precision_mean']:.4f} R={agg['recall_mean']:.4f} "
                  f"AUC={agg['roc_auc_mean']:.4f} params={agg['trainable_params']}",
                  flush=True)

    if args.part in ("windows", "all"):
        print(f"=== Window-size ablation (all 3 features, {args.window_seeds} seeds) ===")
        results["window_ablation"] = []
        for ws in (5, 10, 20, 40):
            runs = [
                run_one(train_df, test_df, FEATURE_SETS["qber+delta+std"], ws, seed=7 + s)
                for s in range(args.window_seeds)
            ]
            agg = aggregate(runs)
            agg["label"] = f"W={ws}"
            results["window_ablation"].append(agg)
            print(f"  W={ws:<3d}              F1={agg['f1_mean']:.4f}+-{agg['f1_std']:.4f} "
                  f"P={agg['precision_mean']:.4f} R={agg['recall_mean']:.4f} "
                  f"AUC={agg['roc_auc_mean']:.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
