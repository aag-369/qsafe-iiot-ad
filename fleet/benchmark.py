"""Measures the fleet correlator's real detection performance across many
random trials: how often does it correctly flag a genuine coordinated
campaign, and how often does it falsely alarm on purely independent,
unrelated per-device activity? Mirrors the spirit of
orchestrator/benchmark.py — real numbers from real runs, not assumed.

Usage:
    python -m fleet.benchmark --trials 15 --devices 6 --rounds 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from crypto_agility.kem_backend import get_kem_backend, is_liboqs_available
from fleet.correlator import FleetCorrelator, FleetCorrelatorConfig
from fleet.simulator import FleetConfig, FleetSimulator
from orchestrator.detector_runner import DetectorRunner
from orchestrator.type_runner import AttackTypeRunner
from qkd_sim.qber_stream_multiclass import ATTACKING_TYPES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--devices", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--qubits", type=int, default=32)
    parser.add_argument("--escalate-threshold", type=float, default=0.86)
    parser.add_argument("--min-devices", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.86)
    parser.add_argument("--out", default="models/fleet_benchmark_report.json")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    detector = DetectorRunner()
    type_runner = AttackTypeRunner()
    backend = get_kem_backend()
    correlator = FleetCorrelator(
        FleetCorrelatorConfig(
            min_devices=args.min_devices,
            confidence_threshold=args.confidence_threshold,
        )
    )
    sim = FleetSimulator(detector, type_runner, backend, correlator)

    t0 = time.time()

    def run_trials(scenario: str, n: int, base_seed: int, **extra) -> list:
        results = []
        for i in range(n):
            cfg = FleetConfig(
                n_devices=args.devices,
                n_rounds=args.rounds,
                n_qubits_per_round=args.qubits,
                scenario=scenario,
                escalate_threshold=args.escalate_threshold,
                seed=base_seed + i,
                **extra,
            )
            results.append(sim.run(cfg))
        return results

    # --- Coordinated campaigns: does the correlator catch them? ---
    coordinated_hits = 0
    coordinated_precisions = []
    coordinated_recalls = []
    coordinated_results = []
    for i, atype in enumerate(ATTACKING_TYPES * ((args.trials // 3) + 1)):
        if i >= args.trials:
            break
        results = run_trials(
            "coordinated_campaign", 1, args.seed + 1000 + i,
            campaign_attack_type=atype, campaign_fraction=0.5,
        )
        result = results[0]
        targets = {d.device_id for d in result.devices if d.is_campaign_target}
        flagged = set()
        for alert in result.correlator_result.alerts:
            flagged |= set(alert.device_ids)

        true_positive_devices = flagged & targets
        false_positive_devices = flagged - targets
        hit = len(true_positive_devices) > 0
        if hit:
            coordinated_hits += 1
        precision = len(true_positive_devices) / len(flagged) if flagged else 0.0
        recall = len(true_positive_devices) / len(targets) if targets else 0.0
        coordinated_precisions.append(precision)
        coordinated_recalls.append(recall)
        coordinated_results.append(
            {
                "attack_type": atype.name.lower(),
                "targets": sorted(targets),
                "flagged": sorted(flagged),
                "hit": hit,
            }
        )

    # --- Independent, uncorrelated activity: does it stay quiet? ---
    independent_results = run_trials("independent_attacks", args.trials, args.seed + 2000)
    independent_false_alarms = sum(1 for r in independent_results if len(r.correlator_result.alerts) > 0)

    # --- Calm baseline: should basically never alert ---
    calm_results = run_trials("calm", args.trials, args.seed + 3000)
    calm_false_alarms = sum(1 for r in calm_results if len(r.correlator_result.alerts) > 0)

    wall_clock = time.time() - t0

    report = {
        "trials_per_scenario": args.trials,
        "n_devices": args.devices,
        "n_rounds": args.rounds,
        "escalate_threshold": args.escalate_threshold,
        "correlator_min_devices": args.min_devices,
        "correlator_confidence_threshold": args.confidence_threshold,
        "liboqs_available": is_liboqs_available(),
        "kem_backend": type(backend).__name__,
        "coordinated_campaign": {
            "recall_any_target_flagged": coordinated_hits / len(coordinated_results),
            "mean_device_precision": sum(coordinated_precisions) / len(coordinated_precisions),
            "mean_device_recall": sum(coordinated_recalls) / len(coordinated_recalls),
            "n_trials": len(coordinated_results),
            "trials": coordinated_results,
        },
        "independent_attacks": {
            "false_alarm_rate": independent_false_alarms / len(independent_results),
            "n_trials": len(independent_results),
        },
        "calm": {
            "false_alarm_rate": calm_false_alarms / len(calm_results),
            "n_trials": len(calm_results),
        },
        "wall_clock_s": wall_clock,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k not in ("coordinated_campaign",)}, indent=2))
    print("coordinated_campaign summary:", {
        k: v for k, v in report["coordinated_campaign"].items() if k != "trials"
    })
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
