# Q-Safe IIoT-AD

**A Crypto-Agile, AI-Gated Post-Quantum Security Framework for Resource-Constrained Industrial IoT Infrastructures**

[![CI](https://github.com/USERNAME/qsafe-iiot-ad/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/qsafe-iiot-ad/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

> Replace `USERNAME` in the badge URL above with your GitHub username once pushed.

IIoT deployments (energy grids, water treatment, industrial control
systems) are still anchored to RSA/ECC, which Shor's algorithm breaks once
a Cryptographically Relevant Quantum Computer (CRQC) exists — including
retroactively, via "Harvest Now, Decrypt Later" (HNDL) interception
happening today. Migrating to NIST post-quantum KEMs is the fix, but
running a hardened PQC profile *all the time* on a Cortex-M4-class edge
device costs real-time latency and battery life the OT environment can't
spare.

Q-Safe IIoT-AD's answer: watch the physical layer (QKD channel QBER) with a
lightweight temporal model, and only pay for the hardened crypto profile
when there's actual evidence of interception. This repo is a complete,
runnable implementation of that idea — real BB84 simulation, a real trained
GRU detector, and real post-quantum KEM operations — not a mockup.

## Architecture

```mermaid
flowchart LR
    subgraph Physical["Physical layer — qkd_sim/"]
        BB84["BB84 QKD simulation\n(Qiskit + Aer)"]
        EVE["Intercept-resend\neavesdropper model"]
        QBER["QBER telemetry stream"]
        BB84 --> QBER
        EVE -.injects.-> BB84
    end

    subgraph AI["Detection layer — ai_detector/"]
        FEAT["Feature windows\n(qber, delta, rolling std)"]
        GRU["Quantized GRU\n(INT8 TFLite)"]
        CONF["Confidence score"]
        QBER --> FEAT --> GRU --> CONF
    end

    subgraph Crypto["Crypto-agility layer — crypto_agility/"]
        SWITCH["Switch controller\n(hysteresis + cooldown)"]
        BIKE["BIKE-L1\n(low-overhead baseline)"]
        HQC["HQC-128\n(hardened profile)"]
        CONF --> SWITCH
        SWITCH -->|escalate| HQC
        SWITCH -->|de-escalate| BIKE
    end

    subgraph Orchestrator["orchestrator/"]
        BENCH["Benchmark harness:\nadaptive vs. always-on-HQC-128"]
    end

    HQC --> BENCH
    BIKE --> BENCH
```

| Layer | Module | What it does |
|---|---|---|
| Physical | `qkd_sim/` | Real Qiskit BB84 circuits (state prep, intercept-resend eavesdropper, sifting) produce a labeled QBER time series with injected "stealthy" HNDL-style attack windows. |
| Detection | `ai_detector/` | A 2-layer GRU (trained with `tf.keras`) classifies each round as benign channel noise vs. active interception; quantized to TFLite for embedded deployment. |
| Crypto-agility | `crypto_agility/` | Wraps real `liboqs` KEM operations for BIKE-L1 and HQC-128, with a hysteresis-based switch controller driven by the detector's confidence score. |
| Orchestration | `orchestrator/` | Runs the full pipeline round-by-round and benchmarks the AI-gated adaptive scheme against a static always-on-HQC-128 baseline. |

## Results (this repo's reproducible run)

These are the actual numbers produced by the code in this repo (see
`models/train_metrics.json` and `models/benchmark_report.json`), not
copied from the paper abstract — regenerate them yourself with the
commands in [Reproducing the results](#reproducing-the-results).

| Metric | Value |
|---|---|
| Detector F1 (held-out QBER stream) | **0.936** (precision 0.937, recall 0.934, ROC-AUC 0.992) |
| Operational F1 (post-hysteresis, real KEM ops) | 0.867 (precision 0.796, recall 0.951) |
| KEM latency, AI-gated adaptive scheme | 3.94 s / 2000 rounds |
| KEM latency, static always-on HQC-128 | 14.67 s / 2000 rounds |
| **CPU/latency reduction from adaptive gating** | **73.2%** |
| Detector trainable parameters | 4,097 |
| Quantized model size (INT8 weights) | 80.6 KB (from 88.6 KB float32) |
| KEM backend used | real `liboqs` (BIKE-L1, HQC-1/HQC-128) |

The operational F1 is lower than the raw detector F1 because the switch
controller adds hysteresis (a cooldown before de-escalating) to avoid
flapping between profiles on every noisy score — a deliberate,
documented precision/stability trade-off, not a bug. See
`crypto_agility/switch_controller.py`.

## Repository layout

```
qsafe-iiot-ad/
├── qkd_sim/            # BB84 simulation (Qiskit) + QBER telemetry generator
├── ai_detector/         # GRU model, feature engineering, training, INT8 quantization
├── crypto_agility/      # liboqs KEM backend (BIKE-L1 / HQC-128) + switch controller
├── orchestrator/        # End-to-end pipeline + adaptive-vs-static benchmark harness
├── firmware_notes/       # ARM Cortex-M4 / TFLite Micro deployment guide
├── tests/               # pytest unit + integration tests (24 tests)
├── scripts/setup_liboqs.sh  # Builds liboqs (BIKE-L1 + HQC-128 only) from source
├── data/                # Generated QBER streams (regenerable, gitignored)
├── models/              # Trained model, quantized TFLite, benchmark reports (committed)
├── .github/workflows/ci.yml
├── Dockerfile
└── requirements.txt
```

## Quickstart

### Option A — Docker (recommended, fully reproducible)

```bash
docker build -t qsafe-iiot-ad .
docker run --rm -it qsafe-iiot-ad bash
# inside the container:
pytest tests/ -v
python -m orchestrator.benchmark --stream data/qber_test.csv
```

### Option B — local Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build liboqs (BIKE-L1 + HQC-128 only — takes ~10s once cmake/ninja are available)
pip install cmake ninja   # only if you don't have them system-wide already
bash scripts/setup_liboqs.sh

pytest tests/ -v
```

## Reproducing the results

```bash
# 1. Generate labeled QBER telemetry (BB84 + injected HNDL-style attacks)
python -m qkd_sim.qber_stream --out data/qber_train.csv --rounds 7000 --qubits 64 --seed 42
python -m qkd_sim.qber_stream --out data/qber_test.csv  --rounds 2000 --qubits 64 --seed 123

# 2. Train + evaluate the GRU detector
python -m ai_detector.train --train data/qber_train.csv --test data/qber_test.csv

# 3. Quantize to INT8 TFLite (Cortex-M4 target)
python -m ai_detector.quantize

# 4. Run the AI-gated-adaptive vs. always-on-HQC-128 benchmark (real liboqs KEM ops)
python -m orchestrator.benchmark --stream data/qber_test.csv
```

Each step writes its artifacts to `data/` and `models/` and prints a
summary; `models/benchmark_report.json` is the final headline-numbers file.

## Design notes and honesty about what's simulated

- **BB84 physics is real**, not mocked: `qkd_sim/bb84.py` builds actual
  Qiskit circuits with Alice's state preparation, an intercept-resend Eve
  (conditioned on a mid-circuit measurement via Aer's dynamic-circuit
  support), a depolarizing noise channel for benign decoherence, and Bob's
  measurement + basis sifting.
- **The GRU is really trained**, not hand-tuned to hit a number: see
  `ai_detector/train.py`. The reported F1 is measured on a held-out stream
  generated with a different random seed and different attack-window count
  than the training stream.
- **The KEM operations are real liboqs**, not stubbed: `crypto_agility/kem_backend.py`
  prefers the actual Open Quantum Safe `liboqs` C library (built via
  `scripts/setup_liboqs.sh`) and only falls back to a clearly-labeled
  (`simulated=True`) timing-accurate stand-in if liboqs isn't built in the
  current environment (e.g. a restrictive CI runner).
- **What is *not* real**: this does not run on physical QKD hardware or an
  actual Cortex-M4 chip. The CPU-latency figures are host-measured (this
  development machine); see `firmware_notes/CORTEX_M4_DEPLOYMENT.md` for
  exactly what changes when porting to real embedded hardware and why the
  *relative* CPU-reduction claim (73.2% here) is the portable part of the
  result, not the absolute millisecond figures.

## Publishing to your GitHub account

This directory is already a git repository with an initial commit. To push
it to your own GitHub:

```bash
cd qsafe-iiot-ad
git remote add origin https://github.com/<your-username>/qsafe-iiot-ad.git
git branch -M main
git push -u origin main
```

(Create the empty repo on GitHub first — `gh repo create qsafe-iiot-ad --public --source=. --remote=origin` does both steps at once if you have the GitHub CLI installed and authenticated.)

## Citation / origin

This implementation accompanies the abstract "Q-Safe IIoT-AD: A
Crypto-Agile, AI-Gated Post-Quantum Security Framework for
Resource-Constrained Industrial IoT Infrastructures." Keywords: Post-Quantum
Cryptography, Industrial IoT Security, Crypto-Agility, Quantum Key
Distribution, BB84 Protocol, QBER Anomaly Detection, Gated Recurrent Unit,
BIKE, HQC, ARM Cortex-M4, Harvest Now Decrypt Later, Critical Infrastructure
Protection.

## License

MIT — see [LICENSE](LICENSE).
