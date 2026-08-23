# Q-Safe Field Link

A live, device-to-device demonstration of **Q-Safe IIoT-AD**. Two phones (or
a phone and a smartwatch) exchange industrial telemetry over a link secured
by post-quantum cryptography that re-keys itself, in front of you, when the
quantum channel shows evidence of interception.

Every layer of the paper's Fig. 1 runs in real time: real Qiskit BB84
circuits, the committed INT8 GRU detector at its published threshold, the
published hysteresis controller, and real `liboqs` BIKE-L1 / HQC-128
operations.

## Start here

| I want to… | Read |
|---|---|
| Run it at an expo tomorrow | **[RUNBOOK.md](RUNBOOK.md)** — setup, the five-minute script, prepared answers, troubleshooting |
| Know exactly what's real | **[ARCHITECTURE_MAPPING.md](ARCHITECTURE_MAPPING.md)** — every component mapped to the paper, and the real/simulated boundary |

## Three commands

```bash
python -m demo.preflight                  # check everything before you present
python -m qsafe_link.run --devices 3      # start the gateway
python -m demo.record_demo                # record the offline fallback
```

## What you get

```
    Phone A  ──────────▶  Q-Safe edge node  ══════▶  Gateway  ──────────▶  Phone B / watch
   "field sensor"        BB84 · GRU · switch     BIKE-L1 / HQC-128        "control room"
                                                  + AES-256-GCM
                                    │
                         Operator console (projector)
              QBER · detector confidence · profile timeline · rekeys · CPU saved
```

- `/node` — a phone becomes an industrial edge node and streams real telemetry
- `/monitor` — a phone or watch receives and decrypts it (`?compact=1` for a watch)
- `/console` — the projector view, with one-tap adversary scenarios
- `/` — QR codes so anyone can join by camera

The adversary controls do not fake telemetry. They set `eve_intercept_prob`
and `channel_error_prob` on the actual BB84 simulation; QBER rises because
of simulated quantum mechanics.

## What this package adds to the repository

`qkd_sim/`, `ai_detector/`, `crypto_agility/`, `orchestrator/` and `fleet/`
are **imported, not modified** — the published results stay reproducible.

| New | Purpose |
|---|---|
| `secure_channel/` | KEM shared secret → HKDF-SHA256 → AES-256-GCM, rekeyed on every profile switch. The published pipeline measures handshakes but discards the secret, so nothing was actually protected; this closes that gap. |
| `qsafe_link/` | Live BB84 channel per device, streaming single-window inference, node state machine, WebSocket gateway, browser clients. |
| `demo/` | Preflight check, scripted recording, evidence pack, runbook. |

One correction *was* made to existing code: the simulated-KEM fallback in
`crypto_agility/kem_backend.py` had its two profiles' reference timings
inverted, which would have reported a negative CPU saving on any machine
without `liboqs`. See ARCHITECTURE_MAPPING.md § 5.

## Tests

```bash
python -m pytest tests/ -q      # 110 tests
```

57 of those are new: `tests/test_secure_channel.py` (replay, tamper,
cross-direction and post-rekey rejection, key separation) and
`tests/test_qsafe_link.py` (including a check that live streaming inference
reaches **identical decisions** to the batch pipeline that produced the
published numbers).
