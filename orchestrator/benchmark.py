"""
End-to-end benchmark: compares the AI-gated adaptive crypto-agile scheme
against a static always-on-HQC-128 baseline over the same QBER telemetry
stream, using real BIKE-L1 / HQC-128 KEM operations (liboqs) for every
round in both scenarios.

Reports:
  - Operational intrusion-detection F1 (post-hysteresis: did the system
    actually escalate to HQC-128 during real attack windows?), alongside
    the raw detector F1 from ai_detector/train.py.
  - Total KEM handshake latency (ms) for adaptive vs. static-HQC128, and
    the percentage reduction. This is a host-measured, ratio-based saving:
    the absolute latency numbers are from this machine, not a Cortex-M4,
    but the *relative* saving is meaningful because it depends only on how
    often the expensive profile is invoked, which is hardware-independent.

Usage:
    python -m orchestrator.benchmark --stream data/qber_test.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from crypto_agility.kem_backend import KEMProfile, get_kem_backend, is_liboqs_available
from crypto_agility.switch_controller import SwitchController
from orchestrator.detector_runner import DetectorRunner
from orchestrator.pipeline import logs_to_dataframe, run_adaptive, run_static_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", default="data/qber_test.csv")
    parser.add_argument("--model", default="models/gru_detector.keras")
    parser.add_argument("--norm-stats", default="models/norm_stats.json")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--escalate-threshold", type=float, default=None)
    parser.add_argument("--de-escalate-threshold", type=float, default=0.5)
    parser.add_argument("--cooldown-rounds", type=int, default=4)
    parser.add_argument("--out", default="models/benchmark_report.json")
    parser.add_argument("--rounds-csv", default="models/benchmark_rounds.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.stream)
    print(f"Loaded {len(df)} rounds from {args.stream} ({df['label'].sum()} labeled attack rounds)")

    with open(Path(args.model).parent / "train_metrics.json") as f:
        train_metrics = json.load(f)
    escalate_threshold = args.escalate_threshold or train_metrics["threshold"]
    de_escalate_threshold = args.de_escalate_threshold or max(0.0, escalate_threshold - 0.2)

    runner = DetectorRunner(model_path=args.model, norm_stats_path=args.norm_stats, window_size=args.window_size)
    t0 = time.time()
    confidences = runner.score_stream(df)
    print(f"Scored stream in {time.time() - t0:.2f}s")

    print(f"liboqs available: {is_liboqs_available()}")
    backend = get_kem_backend()
    print(f"KEM backend: {type(backend).__name__}")

    controller = SwitchController(
        escalate_threshold=escalate_threshold,
        de_escalate_threshold=de_escalate_threshold,
        cooldown_rounds=args.cooldown_rounds,
    )

    print("Running adaptive (AI-gated) scenario...")
    t0 = time.time()
    adaptive_logs = run_adaptive(df, confidences, backend, controller)
    adaptive_elapsed = time.time() - t0
    adaptive_df = logs_to_dataframe(adaptive_logs)

    print("Running static always-on-HQC-128 baseline scenario...")
    t0 = time.time()
    static_logs = run_static_profile(df, backend, KEMProfile.HQC_128)
    static_elapsed = time.time() - t0
    static_df = logs_to_dataframe(static_logs)

    # --- Operational detection performance (post-hysteresis) ---
    y_true = adaptive_df["label"].to_numpy()
    y_pred_escalated = (adaptive_df["profile"] == KEMProfile.HQC_128.value).astype(int).to_numpy()
    op_f1 = f1_score(y_true, y_pred_escalated, zero_division=0)
    op_precision = precision_score(y_true, y_pred_escalated, zero_division=0)
    op_recall = recall_score(y_true, y_pred_escalated, zero_division=0)

    # --- Crypto cost comparison ---
    adaptive_total_ms = float(adaptive_df["total_ms"].sum())
    static_total_ms = float(static_df["total_ms"].sum())
    reduction_pct = (1 - adaptive_total_ms / static_total_ms) * 100

    n_rounds_hqc = int((adaptive_df["profile"] == KEMProfile.HQC_128.value).sum())
    n_rounds_bike = int((adaptive_df["profile"] == KEMProfile.BIKE_L1.value).sum())

    report = {
        "stream": args.stream,
        "n_rounds": len(df),
        "n_attack_rounds": int(df["label"].sum()),
        "liboqs_available": is_liboqs_available(),
        "kem_backend": type(backend).__name__,
        "escalate_threshold": escalate_threshold,
        "de_escalate_threshold": de_escalate_threshold,
        "cooldown_rounds": args.cooldown_rounds,
        "detector_f1_from_training": train_metrics["f1"],
        "operational_f1": op_f1,
        "operational_precision": op_precision,
        "operational_recall": op_recall,
        "adaptive_total_kem_latency_ms": adaptive_total_ms,
        "static_hqc128_total_kem_latency_ms": static_total_ms,
        "cpu_latency_reduction_pct": reduction_pct,
        "rounds_on_bike_l1_baseline": n_rounds_bike,
        "rounds_on_hqc128_hardened": n_rounds_hqc,
        "wall_clock_adaptive_scenario_s": adaptive_elapsed,
        "wall_clock_static_scenario_s": static_elapsed,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    adaptive_df.to_csv(args.rounds_csv, index=False)

    print("\n=== Q-Safe IIoT-AD Benchmark Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"\nSaved -> {args.out}")
    print(f"Saved per-round log -> {args.rounds_csv}")


if __name__ == "__main__":
    main()
