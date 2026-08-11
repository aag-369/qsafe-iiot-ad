"""
Generates a labeled QBER telemetry time series by repeatedly invoking the
BB84 simulation, and injects "Harvest Now, Decrypt Later" (HNDL) style
intercept-resend attack windows with randomized, partial (stealthy)
interception rates so that naive static thresholding is insufficient and a
temporal model (the GRU in ai_detector/) is genuinely needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bb84 import simulate_bb84_round


@dataclass
class StreamConfig:
    n_rounds: int = 4000
    n_qubits_per_round: int = 64
    # Benign channel noise fluctuates around this baseline (photonic loss /
    # environmental decoherence), independently each round.
    benign_error_mean: float = 0.02
    benign_error_std: float = 0.008
    benign_error_clip: tuple[float, float] = (0.0, 0.06)
    # Attack windows: contiguous stretches of rounds where an eavesdropper is
    # active. Intercept probability is randomized *per window* within this
    # range to include both aggressive and stealthy (low-rate, HNDL-style)
    # interception.
    n_attack_windows: int = 18
    attack_len_range: tuple[int, int] = (15, 60)
    attack_intercept_range: tuple[float, float] = (0.08, 0.65)
    seed: int = 42


@dataclass
class GeneratedStream:
    df: pd.DataFrame
    config: StreamConfig = field(repr=False)


class QBERStreamGenerator:
    """Builds a labeled QBER time series for training/evaluating the GRU."""

    def __init__(self, config: StreamConfig | None = None):
        self.config = config or StreamConfig()

    def _sample_attack_windows(self, rng: np.random.Generator) -> np.ndarray:
        cfg = self.config
        is_attack = np.zeros(cfg.n_rounds, dtype=bool)
        intercept_prob = np.zeros(cfg.n_rounds, dtype=float)

        attempts = 0
        placed = 0
        while placed < cfg.n_attack_windows and attempts < cfg.n_attack_windows * 20:
            attempts += 1
            length = int(rng.integers(cfg.attack_len_range[0], cfg.attack_len_range[1] + 1))
            start = int(rng.integers(0, max(1, cfg.n_rounds - length)))
            end = start + length
            if is_attack[start:end].any():
                continue  # avoid overlapping windows
            window_prob = rng.uniform(*cfg.attack_intercept_range)
            is_attack[start:end] = True
            intercept_prob[start:end] = window_prob
            placed += 1

        return is_attack, intercept_prob

    def generate(self, progress_every: int = 500, verbose: bool = True) -> GeneratedStream:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        is_attack, intercept_prob = self._sample_attack_windows(rng)

        records = []
        t0 = time.time()
        for t in range(cfg.n_rounds):
            benign_err = float(
                np.clip(
                    rng.normal(cfg.benign_error_mean, cfg.benign_error_std),
                    cfg.benign_error_clip[0],
                    cfg.benign_error_clip[1],
                )
            )
            result = simulate_bb84_round(
                n_qubits=cfg.n_qubits_per_round,
                channel_error_prob=benign_err,
                eve_intercept_prob=float(intercept_prob[t]),
                rng=rng,
            )
            records.append(
                {
                    "t": t,
                    "qber": result.qber,
                    "sifted_key_length": result.sifted_key_length,
                    "n_intercepted": result.n_intercepted,
                    "label": int(is_attack[t]),
                }
            )
            if verbose and (t + 1) % progress_every == 0:
                elapsed = time.time() - t0
                rate = (t + 1) / elapsed
                eta = (cfg.n_rounds - t - 1) / rate
                print(f"  [{t + 1}/{cfg.n_rounds}] {rate:.1f} rounds/s, ETA {eta:.0f}s")

        df = pd.DataFrame.from_records(records)
        return GeneratedStream(df=df, config=cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate QBER telemetry dataset")
    parser.add_argument("--out", default="data/qber_stream.csv")
    parser.add_argument("--rounds", type=int, default=4000)
    parser.add_argument("--qubits", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = StreamConfig(n_rounds=args.rounds, n_qubits_per_round=args.qubits, seed=args.seed)
    gen = QBERStreamGenerator(cfg)
    stream = gen.generate()
    stream.df.to_csv(args.out, index=False)
    print(f"Wrote {len(stream.df)} rows to {args.out}")
    print(f"Attack rounds: {stream.df['label'].sum()} / {len(stream.df)}")
