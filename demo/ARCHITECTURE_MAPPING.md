# Q-Safe Field Link — architecture mapping

How the live demonstration maps onto the architecture in the paper
(Fig. 1, § System Architecture), and exactly which parts are real.

---

## 1. Every layer, and where it runs

The paper describes four per-node layers closing a control loop every QKD
round, plus two additive layers. The demo runs all six. Nothing in
`qkd_sim/`, `ai_detector/`, `crypto_agility/`, `orchestrator/` or `fleet/`
was modified to make the demo work — `qsafe_link/` imports them.

| Paper layer | Symbol | Paper module | Demo component | What changed |
|---|---|---|---|---|
| Physical | `q_t` | `qkd_sim/bb84.py` | `qsafe_link/channel.py` → `LiveQKDChannel` | Nothing. Calls `simulate_bb84_round` unmodified, once per tick instead of once per CSV row. |
| Detection | `c_t` | `ai_detector/` | `qsafe_link/detector.py` → `LiveDetector` | Runs the **INT8 TFLite** artifact rather than the float Keras model, and scores one window at a time. Features come from `ai_detector.features.build_windows` — not a re-implementation. |
| Crypto-agility | `P_t` | `crypto_agility/switch_controller.py` | Used directly by `qsafe_link/node.py` | Nothing. Same class, same published thresholds. |
| Orchestration | — | `orchestrator/pipeline.py` | `qsafe_link/node.py` → `EdgeNode.step()` | **This is the one real extension.** See § 2. |
| Attack type | `a_t` | `orchestrator/type_runner.py` | `qsafe_link/detector.py` → `LiveTypeTagger` | Batched across devices at 1/8th the round rate. Still never consulted by the switch. |
| Fleet correlation | — | `fleet/correlator.py` | `qsafe_link/runtime.py` → `_run_fleet_correlation` | Nothing. Same `FleetCorrelator`, fed live arrays. |

## 2. The one thing the demo adds: `secure_channel/`

`orchestrator/pipeline.py` performs a full KEM handshake every round and
**discards the shared secret**. That is correct for a cost benchmark — the
measurement is the point — but it means no application data is ever
protected, so there is no link for two devices to talk over.

`secure_channel/` closes that gap:

```
KEM shared secret ──HKDF-SHA256──> 2 × AES-256 traffic keys ──> AES-256-GCM frames
        ▲                                                              │
        └────────── re-derived on every profile switch ────────────────┘
```

- `session.py` — handshake, key schedule, and live rekey. The HKDF `info`
  string binds protocol version, session id, epoch and KEM mechanism, so
  keys derived under BIKE-L1 can never collide with keys derived under
  HQC-128 even from an identical shared secret.
- `aead.py` — AES-256-GCM with `epoch‖seq` nonces, separate keys per
  direction, and a 64-frame sliding replay window. Epoch, direction and
  sequence are bound into the AAD, so a frame cannot be replayed into the
  other direction or under a different epoch header.
- `metrics.py` — two cost figures, deliberately kept distinct (§ 4).

**Why rekey on switch rather than per round.** A production link does not
re-handshake every packet. Rekeying on profile change is both more
realistic and what makes the switch *observable*: the session key
fingerprint visibly changes on every screen at the moment the gate fires,
because a real HQC-128 handshake just ran.

27 tests cover this layer (`tests/test_secure_channel.py`), including
replay rejection, tamper rejection, cross-direction replay, post-rekey
frame rejection, and the case where a forged high-sequence frame must not
be able to shift the replay window and lock out legitimate traffic.

## 3. What is real, and what is not

Stated plainly, because the paper does the same and a demo that overclaims
is worse than no demo.

### Real

- **BB84 physics.** Actual Qiskit circuits: Alice's state preparation, an
  intercept-resend Eve conditioned on a mid-circuit measurement via Aer's
  dynamic-circuit support, a depolarizing noise channel, Bob's measurement,
  and basis sifting. QBER is computed from the sifted key.
- **The adversary controls.** Every scenario button sets
  `eve_intercept_prob` / `channel_error_prob` on that real simulation. QBER
  rises because of simulated quantum mechanics. **No QBER value anywhere in
  this demo is synthesised directly.**
- **The detector.** The committed 4,097-parameter GRU, quantized to INT8,
  at the committed threshold (0.86) and window (20). Verified to agree with
  `orchestrator/detector_runner.py` on 100% of decisions over a 200-round
  replay of `data/qber_test.csv`.
- **The KEM operations.** Real `liboqs` BIKE-L1 and HQC-1 where the library
  is built. Keygen, encapsulation and decapsulation, with the shared secret
  checked for agreement before any traffic key is installed.
- **The AEAD.** Real AES-256-GCM via `cryptography`. The ciphertext shown
  on screen is the ciphertext that crossed the link.
- **The application traffic.** Telemetry the phone actually produced, from
  touch input and — where the browser allows it — the accelerometer.

### Not real, and labelled as such

- **No QKD hardware.** There is no photon source. BB84 is simulated at
  circuit level, exactly as in the paper.
- **No Cortex-M4.** Timings are host-measured. The paper makes the same
  distinction: the portable claim is the *ratio* between profiles, not the
  absolute milliseconds. See `firmware_notes/CORTEX_M4_DEPLOYMENT.md`.
- **The phone is not the PQC endpoint.** There is no browser-side BIKE-L1
  (only HQC-128 exists in WASM, via PQClean), so rather than half-implement
  it, the phone is the node's sensor and HMI over the LAN, and the
  PQC-protected hop is node ↔ gateway. This matches how an industrial
  gateway actually terminates crypto for field devices.
- **The simulated KEM fallback.** If `liboqs` is not built, a
  timing-calibrated stand-in is used and every result carries
  `simulated=True`. The UI shows an amber pill saying so; the recorded
  report says so in its `provenance` block.

## 4. The two cost figures

Reporting one number here would misrepresent the result, so `LinkMetrics`
tracks both:

| Figure | Question it answers | How it is computed |
|---|---|---|
| `paper_equivalent_saved_pct` | *What does the paper's methodology give on this live session?* | One KEM handshake charged per round at the active profile, versus one HQC-128 handshake per round — identical to `orchestrator/pipeline.py`. **This is the figure to quote next to the published 73.2%.** |
| `cpu_saved_pct` | *What did this link actually spend?* | Real rekey costs, against a counterfactual charged one hardened handshake per rekey that actually happened. Deliberately conservative: it does not invent handshakes the adaptive policy avoided. |

Both are priced with per-profile handshake costs **measured on the host at
startup**, not from a table of constants. Measured here: BIKE-L1 ≈ 0.78 ms,
HQC-128 ≈ 9.22 ms — a ratio of about 11.8×.

## 5. A correction made to the existing code

`crypto_agility/kem_backend.py` carried reference timings for its simulated
fallback that had the two profiles **inverted** — BIKE-L1 at 1.85 ms and
HQC-128 at 0.47 ms, i.e. the "low-overhead baseline" priced at roughly four
times the "hardened" profile.

This never affected the published results: `models/benchmark_report.json`
records `liboqs_available: true`, and back-solving it gives ≈ 0.78 ms for
BIKE-L1 and ≈ 7.34 ms for HQC-128, consistent with the real library. But on
any machine without `liboqs` built, the live benchmark would have reported a
**negative** CPU saving — inverting the framework's central claim.

The table is now calibrated to this project's own measured liboqs values,
with the provenance recorded in a comment at the definition site.

## 6. Calibration: qubits per round

The detector was trained on 64-qubit rounds. Rounds with fewer qubits
produce a shorter sifted key, so a single bit error moves QBER further, the
rolling-σ feature spikes, and spurious escalations rise sharply:

| qubits/round | benign QBER σ | spurious escalations |
|---|---|---|
| 32 | 0.0317 | 5.0% of rounds |
| 48 | 0.0249 | 0.2% |
| **64 (training distribution)** | **0.0225** | **0.07%** |
| 96 | 0.0169 | 0.00% |

Reference: the training/test stream's benign σ is 0.0228. **The demo
defaults to 64**, matching it.

The residual 0.07% — roughly one spurious escalation per 8 minutes per
device, 0.45% of time on the hardened profile — is consistent with the
paper's operational precision of 0.796. It is expected behaviour, not a
defect, and it fails toward *more* security rather than less. See
`RUNBOOK.md` § "If it escalates with no attack running".
