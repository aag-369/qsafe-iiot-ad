"""
Q-Safe IIoT-AD — web backend.

A FastAPI service that wraps the *actual* project modules (qkd_sim,
ai_detector, crypto_agility, orchestrator) so the website's "live demo" is
a real BB84 simulation feeding a real trained GRU detector feeding a real
liboqs-backed crypto-agile switch — the same code exercised by
`orchestrator/benchmark.py` and `tests/`, not a re-implementation.

Run:
    uvicorn web.backend.app:app --reload --port 8000
(run from the repository root so the `qkd_sim` / `ai_detector` / ... top
-level packages are importable)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ai_detector.features import WindowConfig, build_windows, normalize_features  # noqa: E402
from crypto_agility.kem_backend import KEMProfile, get_kem_backend, is_liboqs_available  # noqa: E402
from crypto_agility.switch_controller import SwitchController  # noqa: E402
from orchestrator.detector_runner import DetectorRunner  # noqa: E402
from orchestrator.pipeline import logs_to_dataframe, run_adaptive, run_static_profile  # noqa: E402
from qkd_sim.bb84 import simulate_bb84_round  # noqa: E402
from qkd_sim.qber_stream import QBERStreamGenerator, StreamConfig  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Q-Safe IIoT-AD API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lazy-loaded, process-wide singletons -----------------------------------
_detector_runner: DetectorRunner | None = None


def get_detector() -> DetectorRunner:
    global _detector_runner
    if _detector_runner is None:
        _detector_runner = DetectorRunner(
            model_path=str(MODELS_DIR / "gru_detector.keras"),
            norm_stats_path=str(MODELS_DIR / "norm_stats.json"),
            window_size=20,
        )
    return _detector_runner


def _load_threshold() -> float:
    with open(MODELS_DIR / "train_metrics.json") as f:
        return json.load(f)["threshold"]


# --- Request/response schemas -------------------------------------------------
class ProbeRequest(BaseModel):
    n_qubits: int = Field(64, ge=8, le=512)
    channel_error_prob: float = Field(0.02, ge=0.0, le=0.3)
    eve_intercept_prob: float = Field(0.0, ge=0.0, le=1.0)


class LiveSimRequest(BaseModel):
    n_rounds: int = Field(80, ge=20, le=300)
    n_qubits_per_round: int = Field(48, ge=16, le=128)
    inject_attack: bool = True
    attack_intensity: float = Field(0.35, ge=0.05, le=0.9)
    seed: int | None = None


class BenchmarkRequest(BaseModel):
    n_rounds: int = Field(120, ge=20, le=400)
    n_qubits_per_round: int = Field(48, ge=16, le=128)
    seed: int | None = None


# --- Static content -----------------------------------------------------------
PROJECT_INFO = {
    "title": "Q-Safe IIoT-AD",
    "subtitle": "A Crypto-Agile, AI-Gated Post-Quantum Security Framework for Resource-Constrained Industrial IoT Infrastructures",
    "abstract": (
        "IIoT deployments across energy grids, water treatment, and industrial control systems remain "
        "anchored to RSA/ECC, which Shor's algorithm breaks once a Cryptographically Relevant Quantum "
        "Computer exists. Q-Safe IIoT-AD watches the physical-layer QKD channel (QBER) with a lightweight "
        "quantized GRU, and only escalates from a low-overhead BIKE-L1 profile to a hardened HQC-128 "
        "profile when there is real evidence of interception — including stealthy 'Harvest Now, Decrypt "
        "Later' reconnaissance."
    ),
    "keywords": [
        "Post-Quantum Cryptography", "Industrial IoT Security", "Crypto-Agility",
        "Quantum Key Distribution", "BB84 Protocol", "QBER Anomaly Detection",
        "Gated Recurrent Unit", "BIKE", "HQC", "ARM Cortex-M4",
        "Harvest Now Decrypt Later", "Critical Infrastructure Protection",
    ],
    "pipeline": [
        {
            "stage": "Physical Layer",
            "module": "qkd_sim/",
            "description": "Real Qiskit BB84 circuits — state preparation, an intercept-resend eavesdropper conditioned on a mid-circuit measurement, and a depolarizing noise channel — produce a continuous QBER telemetry stream.",
        },
        {
            "stage": "AI Detection",
            "module": "ai_detector/",
            "description": "A quantized 2-layer GRU classifies each round as benign channel noise vs. active interception from a rolling window of QBER, its delta, and its rolling standard deviation.",
        },
        {
            "stage": "Crypto-Agility",
            "module": "crypto_agility/",
            "description": "A hysteresis-based switch controller escalates real liboqs KEM operations from BIKE-L1 (low-overhead baseline) to HQC-128 (hardened, IND-CCA2) only when the detector's confidence crosses a calibrated threshold.",
        },
        {
            "stage": "Orchestration",
            "module": "orchestrator/",
            "description": "Runs the full pipeline round-by-round and benchmarks the AI-gated adaptive scheme against a static always-on-HQC-128 baseline using real KEM handshake timings.",
        },
    ],
}


# --- Endpoints -----------------------------------------------------------------
@app.get("/api/health")
def health():
    detector_ok = (MODELS_DIR / "gru_detector.keras").exists()
    return {
        "status": "ok",
        "liboqs_available": is_liboqs_available(),
        "kem_backend": type(get_kem_backend()).__name__,
        "detector_model_present": detector_ok,
    }


@app.get("/api/project-info")
def project_info():
    return PROJECT_INFO


@app.get("/api/results/summary")
def results_summary():
    """Precomputed, reproducible results from the committed training run and
    benchmark — loads instantly, no live compute needed for the dashboard."""
    try:
        with open(MODELS_DIR / "train_metrics.json") as f:
            train_metrics = json.load(f)
        with open(MODELS_DIR / "benchmark_report.json") as f:
            benchmark = json.load(f)
        with open(MODELS_DIR / "gru_detector_int8_size_report.json") as f:
            size_report = json.load(f)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Missing results artifact: {e.filename}")

    return {
        "train_metrics": train_metrics,
        "benchmark": benchmark,
        "quantization": size_report,
    }


@app.post("/api/simulate/probe")
def simulate_probe(req: ProbeRequest):
    """Single BB84 round — the 'try it yourself' knob-turning widget."""
    rng = np.random.default_rng()
    t0 = time.perf_counter()
    result = simulate_bb84_round(
        n_qubits=req.n_qubits,
        channel_error_prob=req.channel_error_prob,
        eve_intercept_prob=req.eve_intercept_prob,
        rng=rng,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "qber": result.qber,
        "sifted_key_length": result.sifted_key_length,
        "n_qubits": result.n_qubits,
        "n_intercepted": result.n_intercepted,
        "simulation_time_ms": elapsed_ms,
    }


@app.post("/api/simulate/live")
def simulate_live(req: LiveSimRequest):
    """Full pipeline on a fresh, real BB84-generated stream: QKD -> GRU
    detector -> crypto-agile switch. Every point returned here was actually
    computed, not canned."""
    seed = req.seed if req.seed is not None else int(time.time() * 1000) % (2**31)

    n_windows = max(3, req.n_rounds // 12) if req.inject_attack else 0
    cfg = StreamConfig(
        n_rounds=req.n_rounds,
        n_qubits_per_round=req.n_qubits_per_round,
        n_attack_windows=n_windows,
        attack_len_range=(6, max(7, req.n_rounds // 6)),
        attack_intercept_range=(
            max(0.05, req.attack_intensity - 0.15),
            min(0.95, req.attack_intensity + 0.15),
        ),
        seed=seed,
    )
    gen = QBERStreamGenerator(cfg)
    stream = gen.generate(verbose=False)
    df = stream.df

    detector = get_detector()
    confidences = detector.score_stream(df)

    threshold = _load_threshold()
    controller = SwitchController(
        escalate_threshold=threshold,
        de_escalate_threshold=max(0.05, threshold - 0.36),
        cooldown_rounds=4,
    )

    points = []
    for row, conf in zip(df.itertuples(), confidences):
        decision = controller.step(int(row.t), float(conf))
        points.append(
            {
                "t": int(row.t),
                "qber": float(row.qber),
                "label": int(row.label),
                "confidence": float(conf),
                "profile": decision.profile.value,
                "escalated": bool(decision.escalated_this_round),
            }
        )

    return {
        "seed": seed,
        "n_rounds": req.n_rounds,
        "n_qubits_per_round": req.n_qubits_per_round,
        "threshold": threshold,
        "points": points,
    }


@app.post("/api/benchmark/run")
def benchmark_run(req: BenchmarkRequest):
    """Live adaptive-vs-static-HQC-128 benchmark using real liboqs KEM
    operations for every round in both scenarios — a smaller/faster version
    of orchestrator/benchmark.py sized for a web request."""
    seed = req.seed if req.seed is not None else 7

    cfg = StreamConfig(
        n_rounds=req.n_rounds,
        n_qubits_per_round=req.n_qubits_per_round,
        n_attack_windows=max(2, req.n_rounds // 25),
        attack_len_range=(6, 20),
        attack_intercept_range=(0.1, 0.6),
        seed=seed,
    )
    gen = QBERStreamGenerator(cfg)
    stream = gen.generate(verbose=False)
    df = stream.df

    detector = get_detector()
    confidences = detector.score_stream(df)

    threshold = _load_threshold()
    backend = get_kem_backend()

    controller = SwitchController(
        escalate_threshold=threshold,
        de_escalate_threshold=max(0.05, threshold - 0.36),
        cooldown_rounds=4,
    )
    t0 = time.time()
    adaptive_logs = run_adaptive(df, confidences, backend, controller)
    adaptive_elapsed = time.time() - t0
    adaptive_df = logs_to_dataframe(adaptive_logs)

    t0 = time.time()
    static_logs = run_static_profile(df, backend, KEMProfile.HQC_128)
    static_elapsed = time.time() - t0
    static_df = logs_to_dataframe(static_logs)

    from sklearn.metrics import f1_score, precision_score, recall_score

    y_true = adaptive_df["label"].to_numpy()
    y_pred = (adaptive_df["profile"] == KEMProfile.HQC_128.value).astype(int).to_numpy()

    adaptive_total_ms = float(adaptive_df["total_ms"].sum())
    static_total_ms = float(static_df["total_ms"].sum())

    return {
        "n_rounds": req.n_rounds,
        "n_attack_rounds": int(df["label"].sum()),
        "kem_backend": type(backend).__name__,
        "liboqs_available": is_liboqs_available(),
        "operational_f1": f1_score(y_true, y_pred, zero_division=0),
        "operational_precision": precision_score(y_true, y_pred, zero_division=0),
        "operational_recall": recall_score(y_true, y_pred, zero_division=0),
        "adaptive_total_kem_latency_ms": adaptive_total_ms,
        "static_hqc128_total_kem_latency_ms": static_total_ms,
        "cpu_latency_reduction_pct": (1 - adaptive_total_ms / static_total_ms) * 100 if static_total_ms else 0,
        "rounds_on_bike_l1": int((adaptive_df["profile"] == KEMProfile.BIKE_L1.value).sum()),
        "rounds_on_hqc128": int((adaptive_df["profile"] == KEMProfile.HQC_128.value).sum()),
        "wall_clock_adaptive_s": adaptive_elapsed,
        "wall_clock_static_s": static_elapsed,
    }


# --- Static frontend ------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
