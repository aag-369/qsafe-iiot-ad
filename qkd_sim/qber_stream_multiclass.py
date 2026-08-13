"""Multi-class extension of qber_stream.py: generates a labeled QBER
telemetry stream that distinguishes *which kind* of attack is happening,
not just "attack vs. benign".

This module deliberately does NOT modify qber_stream.py or bb84.py — the
original binary detector's training data, and every existing test that
depends on it, stay byte-for-byte reproducible. Instead, this module reuses
`simulate_bb84_round`'s two existing parameters (`channel_error_prob`,
`eve_intercept_prob`) with different value ranges per attack type, so no new
quantum-circuit logic is introduced:

  * EAVESDROP — the original intercept-resend attack (elevated
    eve_intercept_prob, benign channel noise otherwise). Textbook BB84
    eavesdropping: intercepting and resending a qubit introduces a ~25%
    error rate on the intercepted fraction, so QBER rises but stays
    moderate and structured.
  * JAMMING — a channel-disruption / denial-of-service style attack:
    directly elevated depolarizing channel noise (no interception at all).
    QBER spikes much higher and noisier than eavesdropping — a "loud",
    easy-to-notice signature, but a different one.
  * PNS — a stealthy, photon-number-splitting-style attack: only a small,
    sustained bias above the benign noise floor. Real PNS attacks exploit
    multi-photon pulses without necessarily causing the same disturbance as
    full intercept-resend, so they are the intentionally hard class here —
    this is what motivates a temporal model over a naive QBER threshold.

The switch controller's escalate/de-escalate decision keeps using the
original binary detector unchanged; this module only feeds the *additional*,
additive attack-type classifier used for reporting and fleet correlation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd

from .bb84 import simulate_bb84_round


class AttackType(IntEnum):
    BENIGN = 0
    EAVESDROP = 1
    JAMMING = 2
    PNS = 3


ATTACK_TYPE_NAMES = {
    AttackType.BENIGN: "benign",
    AttackType.EAVESDROP: "eavesdrop",
    AttackType.JAMMING: "jamming",
    AttackType.PNS: "pns",
}

# Per-type (channel_error_prob range, eve_intercept_prob range) used to
# parameterize simulate_bb84_round for rounds inside an attack window of
# that type. Ranges are sampled once per window (not per round) so each
# window has one coherent "personality", matching how a real, sustained
# attack episode would behave rather than jittering wildly round to round.
ATTACK_PROFILES: dict[AttackType, dict[str, tuple[float, float]]] = {
    AttackType.EAVESDROP: {
        "channel_error_prob_range": (0.015, 0.03),
        "eve_intercept_prob_range": (0.08, 0.65),
    },
    AttackType.JAMMING: {
        "channel_error_prob_range": (0.15, 0.45),
        "eve_intercept_prob_range": (0.0, 0.0),
    },
    AttackType.PNS: {
        "channel_error_prob_range": (0.032, 0.058),
        "eve_intercept_prob_range": (0.0, 0.05),
    },
}

ATTACKING_TYPES = (AttackType.EAVESDROP, AttackType.JAMMING, AttackType.PNS)


@dataclass
class MultiClassStreamConfig:
    n_rounds: int = 6000
    n_qubits_per_round: int = 64
    benign_error_mean: float = 0.02
    benign_error_std: float = 0.008
    benign_error_clip: tuple[float, float] = (0.0, 0.06)
    # Random-placement mode: this many non-overlapping windows are scattered
    # through the stream, each independently assigned one of ATTACKING_TYPES.
    n_attack_windows: int = 18
    attack_len_range: tuple[int, int] = (15, 60)
    # Explicit-placement mode (used by the fleet simulator to inject a
    # specific, time-aligned event across several devices at once): a list
    # of (start, length, AttackType) tuples. When given, these are used
    # *instead of* the random n_attack_windows placement.
    forced_windows: list[tuple[int, int, AttackType]] | None = None
    seed: int = 42


@dataclass
class GeneratedMultiClassStream:
    df: pd.DataFrame
    config: MultiClassStreamConfig = field(repr=False)


class MultiClassQBERStreamGenerator:
    """Builds a labeled QBER time series with per-round attack-type ground
    truth, for training/evaluating the attack-type classifier and for
    driving fleet simulation scenarios."""

    def __init__(self, config: MultiClassStreamConfig | None = None):
        self.config = config or MultiClassStreamConfig()

    def _sample_random_windows(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Places non-overlapping windows and samples ONE intensity per
        window (not per round) so each attack episode has a single, coherent
        "personality" — a stable elevated level the GRU can actually learn a
        temporal shape from, rather than round-to-round noise."""
        cfg = self.config
        attack_type = np.zeros(cfg.n_rounds, dtype=np.int64)  # AttackType.BENIGN
        chan_override = np.full(cfg.n_rounds, np.nan)
        eve_override = np.zeros(cfg.n_rounds, dtype=float)

        attempts = 0
        placed = 0
        occupied = np.zeros(cfg.n_rounds, dtype=bool)
        while placed < cfg.n_attack_windows and attempts < cfg.n_attack_windows * 20:
            attempts += 1
            length = int(rng.integers(cfg.attack_len_range[0], cfg.attack_len_range[1] + 1))
            start = int(rng.integers(0, max(1, cfg.n_rounds - length)))
            end = start + length
            if occupied[start:end].any():
                continue
            atype = ATTACKING_TYPES[int(rng.integers(0, len(ATTACKING_TYPES)))]
            profile = ATTACK_PROFILES[atype]
            lo, hi = profile["channel_error_prob_range"]
            window_chan = float(rng.uniform(lo, hi))
            lo_e, hi_e = profile["eve_intercept_prob_range"]
            window_eve = float(rng.uniform(lo_e, hi_e)) if hi_e > 0 else 0.0

            occupied[start:end] = True
            attack_type[start:end] = int(atype)
            chan_override[start:end] = window_chan
            eve_override[start:end] = window_eve
            placed += 1

        return attack_type, chan_override, eve_override

    def _apply_forced_windows(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        attack_type = np.zeros(cfg.n_rounds, dtype=np.int64)
        chan_override = np.full(cfg.n_rounds, np.nan)
        eve_override = np.zeros(cfg.n_rounds, dtype=float)
        for start, length, atype in cfg.forced_windows:
            end = min(cfg.n_rounds, start + length)
            start = max(0, start)
            profile = ATTACK_PROFILES[atype]
            lo, hi = profile["channel_error_prob_range"]
            window_chan = float(rng.uniform(lo, hi))
            lo_e, hi_e = profile["eve_intercept_prob_range"]
            window_eve = float(rng.uniform(lo_e, hi_e)) if hi_e > 0 else 0.0
            attack_type[start:end] = int(atype)
            chan_override[start:end] = window_chan
            eve_override[start:end] = window_eve
        return attack_type, chan_override, eve_override

    def generate(self, progress_every: int = 500, verbose: bool = True) -> GeneratedMultiClassStream:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        if cfg.forced_windows is not None:
            attack_type, chan_override, eve_override = self._apply_forced_windows(rng)
        else:
            attack_type, chan_override, eve_override = self._sample_random_windows(rng)

        records = []
        t0 = time.time()
        for t in range(cfg.n_rounds):
            atype = AttackType(attack_type[t])
            benign_err = float(
                np.clip(
                    rng.normal(cfg.benign_error_mean, cfg.benign_error_std),
                    cfg.benign_error_clip[0],
                    cfg.benign_error_clip[1],
                )
            )
            if atype == AttackType.BENIGN:
                channel_error_prob = benign_err
                eve_intercept_prob = 0.0
            else:
                # Use this window's fixed sampled intensity; add a small
                # amount of round-to-round jitter around it so it still
                # looks like a physical channel, not a step function.
                channel_error_prob = float(
                    np.clip(chan_override[t] + rng.normal(0, 0.004), 0.0, 0.6)
                )
                eve_intercept_prob = float(
                    np.clip(eve_override[t] + rng.normal(0, 0.02), 0.0, 1.0)
                ) if eve_override[t] > 0 else 0.0

            result = simulate_bb84_round(
                n_qubits=cfg.n_qubits_per_round,
                channel_error_prob=channel_error_prob,
                eve_intercept_prob=eve_intercept_prob,
                rng=rng,
            )
            records.append(
                {
                    "t": t,
                    "qber": result.qber,
                    "sifted_key_length": result.sifted_key_length,
                    "n_intercepted": result.n_intercepted,
                    "label": int(atype != AttackType.BENIGN),
                    "attack_type": int(atype),
                }
            )
            if verbose and (t + 1) % progress_every == 0:
                elapsed = time.time() - t0
                rate = (t + 1) / elapsed
                eta = (cfg.n_rounds - t - 1) / rate
                print(f"  [{t + 1}/{cfg.n_rounds}] {rate:.1f} rounds/s, ETA {eta:.0f}s")

        df = pd.DataFrame.from_records(records)
        return GeneratedMultiClassStream(df=df, config=cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate multi-class (attack-type) QBER telemetry dataset")
    parser.add_argument("--out", default="data/qber_multiclass_stream.csv")
    parser.add_argument("--rounds", type=int, default=6000)
    parser.add_argument("--qubits", type=int, default=64)
    parser.add_argument("--windows", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = MultiClassStreamConfig(
        n_rounds=args.rounds,
        n_qubits_per_round=args.qubits,
        n_attack_windows=args.windows,
        seed=args.seed,
    )
    gen = MultiClassQBERStreamGenerator(cfg)
    stream = gen.generate()
    stream.df.to_csv(args.out, index=False)
    print(f"Wrote {len(stream.df)} rows to {args.out}")
    print(stream.df["attack_type"].value_counts().to_dict())
