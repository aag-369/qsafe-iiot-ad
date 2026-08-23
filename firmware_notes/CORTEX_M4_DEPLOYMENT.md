# Deploying to ARM Cortex-M4 class IIoT edge hardware

This repository runs the full Q-Safe IIoT-AD pipeline as a **host-side
simulation** (BB84 via Qiskit/Aer, GRU inference via TensorFlow, KEM
operations via liboqs on x86_64). That is deliberate: it lets every claim in
this project be exercised and measured end-to-end without physical QKD
hardware or a Cortex-M4 board. This document describes the remaining steps
to move the two firmware-relevant components — the quantized detector and
the KEM operations — onto real Cortex-M4 hardware.

## 1. The GRU detector

`ai_detector/quantize.py` produces `models/gru_detector_int8.tflite`, an
INT8-weight-quantized TFLite model. To run it on a Cortex-M4:

1. Use **TensorFlow Lite for Microcontrollers** (TFLite Micro), not full
   TFLite — it has no OS/filesystem/dynamic-allocation dependency and links
   into a bare-metal or RTOS (Zephyr, FreeRTOS, Mbed) firmware image.
2. Enable **CMSIS-NN** kernels (`TFLITE_MICRO` + `CMSIS_NN` build tags) so
   INT8 matrix-multiply/GRU-gate operations use the Cortex-M4's DSP
   extension (SIMD `SMLAD`/`SMLAL` instructions) rather than scalar C.
3. Convert the `.tflite` flatbuffer to a C byte array (`xxd -i` or TFLite
   Micro's `xxd`-equivalent build rule) and link it as `const unsigned char
   g_model[]` in the firmware image.
4. Allocate a `tflite::MicroInterpreter` with a static `tensor_arena`
   (typically 8–32 KB for a model this size — measure with
   `interpreter.arena_used_bytes()` on target).
5. Feed it the same two features computed in `ai_detector/features.py`
   (`qber`, `qber_delta`, `qber_rolling_std` over a rolling 5-sample
   window) computed on-device from the QKD post-processing module's live
   QBER counter — no floating-point QKD simulation needed on-device, only
   the same two scalar features per round.
6. Apply the same normalization constants saved in `models/norm_stats.json`
   (a fixed `(x - mean) / std` computed in firmware as INT8-friendly fixed
   point, or left as float32 input/output as configured in
   `quantize.py`, since Cortex-M4 has a single-precision FPU).

Expected footprint at the current architecture size (32-unit GRU, 16-unit
dense layer, ~2–5K parameters): comfortably under 64 KB flash for
weights+code and under 16 KB RAM for the interpreter arena, i.e. well
within a typical Cortex-M4 IIoT sensor node budget (256 KB–1 MB flash,
64–256 KB RAM is common on STM32F4/NXP LPC54xxx class parts).

## 2. The KEM operations (BIKE-L1 / HQC-128)

`crypto_agility/kem_backend.py` uses `liboqs` compiled for x86_64. For
Cortex-M4 firmware:

1. Cross-compile `liboqs` with `-DOQS_USE_OPENSSL=OFF` (matches
   `scripts/setup_liboqs.sh`) and an ARM cross-toolchain
   (`arm-none-eabi-gcc`), targeting the `portable` (non-AVX) code paths —
   liboqs's BIKE and HQC implementations both ship portable C reference
   variants alongside the x86-SIMD-optimized ones used here.
2. Alternatively, use **wolfSSL** or **mbed TLS**'s PQC forks, which have
   published Cortex-M4 benchmarks for BIKE/HQC and are more commonly used
   in production embedded TLS stacks than linking liboqs directly into
   firmware.
3. `crypto_agility/switch_controller.py`'s decision logic (escalate /
   de-escalate / cooldown) is pure Python here but has no host-specific
   dependency — it is a direct port to C: three comparisons and a counter,
   trivial to reimplement as a firmware state machine calling into
   whichever KEM library is linked.

## 3. What this repo's benchmark numbers do and don't tell you

`orchestrator/benchmark.py` reports **host-measured** KEM latency (real
liboqs operations, real wall-clock time) and a **ratio-based** CPU/latency
reduction (adaptive vs. always-on-HQC-128). The absolute millisecond
figures are from this development machine, not a Cortex-M4 — they will be
larger in absolute terms on an M4 (no AVX/AVX-512, lower clock, no OS
scheduler competing for the core in a well-designed RTOS task, etc.). The
**percentage reduction** is the portable claim: it depends on how much more
expensive HQC-128 is than BIKE-L1 for a given operation mix and how often
the detector escalates — both of which are architecture-independent
properties of the algorithms and the attack scenario, not of the CPU
running them. Validating the absolute-latency claim on real Cortex-M4
silicon is the natural next step once this pipeline is ported per the
sections above.
