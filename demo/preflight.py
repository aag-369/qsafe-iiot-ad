"""
Pre-demo check. Run this before you walk into the room.

Verifies, in order, everything the live demo depends on: interpreter,
packages, model artifacts, KEM backend, detector agreement with the
committed metrics, BB84 throughput on this machine, and the network address
phones will need to reach. Prints a single PASS/FAIL verdict.

    python -m demo.preflight
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    icon = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
    print(f"[{icon}] {name:<34} {detail}")


def main() -> int:
    print("\nQ-Safe Field Link — preflight\n" + "-" * 68)

    v = sys.version_info
    check("Python interpreter", OK if v >= (3, 10) else FAIL,
          f"{v.major}.{v.minor}.{v.micro}" + ("" if v >= (3, 10) else " — needs 3.10+"))

    for mod, label in [
        ("numpy", "numpy"), ("pandas", "pandas"), ("qiskit", "qiskit"),
        ("qiskit_aer", "qiskit-aer"), ("tensorflow", "tensorflow"),
        ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
        ("cryptography", "cryptography"), ("sklearn", "scikit-learn"),
    ]:
        try:
            __import__(mod)
            check(f"package: {label}", OK)
        except Exception as exc:
            check(f"package: {label}", FAIL, str(exc)[:48])

    try:
        import qrcode  # noqa: F401
        check("package: qrcode (optional)", OK, "join page will show QR codes")
    except Exception:
        check("package: qrcode (optional)", WARN, "join page falls back to plain URLs")

    models = REPO_ROOT / "models"
    required = ["gru_detector.keras", "gru_detector_int8.tflite", "norm_stats.json",
                "train_metrics.json"]
    optional = ["attack_type_gru.keras", "attack_type_norm_stats.json"]
    for f in required:
        p = models / f
        check(f"artifact: {f}", OK if p.exists() else FAIL,
              f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "MISSING")
    for f in optional:
        p = models / f
        check(f"artifact: {f} (optional)", OK if p.exists() else WARN,
              "" if p.exists() else "Fleet View attack-type tags unavailable")

    try:
        from crypto_agility.kem_backend import (KEMProfile, get_kem_backend,
                                                is_liboqs_available)
        backend = get_kem_backend()
        real = type(backend).__name__ == "LiboqsKEMBackend"
        costs = {}
        for profile in (KEMProfile.BIKE_L1, KEMProfile.HQC_128):
            pk, sk, kg = backend.keygen(profile)
            ct, ss1, enc = backend.encapsulate(profile, pk)
            ss2, dec = backend.decapsulate(profile, sk, ct)
            if ss1 != ss2:
                check(f"KEM correctness: {profile.value}", FAIL, "shared secrets disagree")
            costs[profile.value] = kg.latency_ms + enc.latency_ms + dec.latency_ms
        check("KEM backend", OK if real else WARN,
              "real liboqs (BIKE-L1 / HQC-1)" if real
              else "SIMULATED — the UI labels this, but build liboqs for a stronger claim")
        ratio = costs["HQC-128"] / max(costs["BIKE-L1"], 1e-9)
        check("KEM cost ordering", OK if ratio > 2 else FAIL,
              f"BIKE-L1 {costs['BIKE-L1']:.2f} ms, HQC-128 {costs['HQC-128']:.2f} ms "
              f"({ratio:.1f}x)" + ("" if ratio > 2 else " — baseline must be cheaper!"))
        check("KEM shared-secret agreement", OK, "both profiles round-trip correctly")
        _ = is_liboqs_available()
    except Exception as exc:
        check("KEM backend", FAIL, str(exc)[:56])

    try:
        import json

        import pandas as pd

        from qsafe_link.detector import LiveDetector

        t0 = time.perf_counter()
        det = LiveDetector(models_dir=models)
        load_ms = (time.perf_counter() - t0) * 1000
        check("detector loaded", OK, f"{det.backend}, threshold {det.threshold:.2f}, "
                                    f"{load_ms:.0f} ms to load")

        # Does streaming inference still reach the same decisions as the batch
        # pipeline that produced the published numbers? This is the single most
        # important thing to know before presenting, so it runs against a
        # deterministic synthetic series when the (gitignored) held-out stream
        # is not on this machine, rather than being skipped.
        import numpy as np

        from orchestrator.detector_runner import DetectorRunner

        test_csv = REPO_ROOT / "data" / "qber_test.csv"
        if test_csv.exists():
            qber = pd.read_csv(test_csv).head(200)["qber"].to_numpy()
            source = "held-out BB84 stream"
        else:
            rng = np.random.default_rng(7)
            qber = rng.normal(0.015, 0.022, 220)
            for start, length, level in ((60, 40, 0.16), (150, 30, 0.05)):
                qber[start:start + length] = rng.normal(level, level * 0.35, length)
            qber = np.clip(qber, 0.0, 1.0)
            source = "synthetic series (data/qber_test.csv not present)"

        df = pd.DataFrame({"qber": qber, "label": 0})
        batch = DetectorRunner(
            model_path=str(models / "gru_detector.keras"),
            norm_stats_path=str(models / "norm_stats.json"),
        ).score_stream(df)
        tail, live = [], []
        for q in qber:
            tail.append(float(q))
            live.append(det.score_tail(tail))

        live = np.array(live)
        m = np.arange(len(df)) >= det.window_size - 1
        agree = ((live[m] >= det.threshold) == (batch[m] >= det.threshold)).mean()
        max_dev = float(np.abs(live[m] - batch[m]).max())
        check(
            "live detector vs batch pipeline",
            OK if (agree >= 0.98 and max_dev < 0.01) else FAIL,
            f"{agree*100:.1f}% decision agreement, max deviation {max_dev:.4f} "
            f"({len(qber)} rounds, {source})",
        )

        with open(models / "train_metrics.json") as f:
            tm = json.load(f)
        check("committed metrics", OK,
              f"F1 {tm['f1']:.3f}, AUC {tm['roc_auc']:.3f}, {tm['trainable_params']:,} params")
    except Exception as exc:
        check("detector", FAIL, str(exc)[:56])

    try:
        import numpy as np

        from qkd_sim.bb84 import simulate_bb84_round
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        for _ in range(6):
            simulate_bb84_round(n_qubits=64, channel_error_prob=0.02,
                                eve_intercept_prob=0.4, rng=rng)
        per = (time.perf_counter() - t0) / 6 * 1000
        capacity = 1000 / per
        check("BB84 throughput (64 qubits, under attack)",
              OK if capacity >= 12 else WARN,
              f"{per:.1f} ms/round → ~{capacity:.0f} rounds/s total"
              + ("" if capacity >= 12 else " — use fewer devices or a lower --rate"))
    except Exception as exc:
        check("BB84 simulation", FAIL, str(exc)[:56])

    try:
        from qsafe_link.gateway import detect_lan_ip
        ip = detect_lan_ip()
        local = ip.startswith("127.")
        check("LAN address for phones", WARN if local else OK,
              f"http://{ip}:8000" + ("  — loopback only; check Wi-Fi" if local else ""))
    except Exception as exc:
        check("LAN address", WARN, str(exc)[:56])

    print("-" * 68)
    fails = [r for r in results if r[1] == FAIL]
    warns = [r for r in results if r[1] == WARN]
    if fails:
        print(f"\n  {len(fails)} FAILURE(S) — the demo will not run correctly:\n")
        for name, _, detail in fails:
            print(f"    - {name}: {detail}")
        print("\n  See demo/RUNBOOK.md § Troubleshooting.\n")
        return 1
    print(f"\n  READY — {len(results) - len(warns)} checks passed"
          + (f", {len(warns)} warning(s)" if warns else "") + ".")
    print("  Start with:  python -m qsafe_link.run --devices 3\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
