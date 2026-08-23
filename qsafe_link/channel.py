"""
A live BB84 quantum channel for one edge node.

This is the part of the demo that must not cheat. The attack controls in the
UI do **not** write a QBER number into a chart -- they set
`eve_intercept_prob` and `channel_error_prob` on a real
`qkd_sim.bb84.simulate_bb84_round` call, which builds an actual Qiskit
circuit with Alice's state preparation, an intercept-resend Eve conditioned
on a mid-circuit measurement, a depolarizing noise channel, and Bob's
measurement and sifting. The QBER the detector then sees rises because of
simulated quantum mechanics, not because someone typed a bigger number.

Attack parameterisation is imported verbatim from
`qkd_sim.qber_stream_multiclass.ATTACK_PROFILES`, the same table used to
generate the data the attack-type classifier was trained on, so a live
"jamming" episode has the same signature as a training-set jamming episode.
No new quantum-circuit logic is introduced anywhere in this module.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from qkd_sim.bb84 import simulate_bb84_round
from qkd_sim.qber_stream_multiclass import ATTACK_PROFILES, ATTACK_TYPE_NAMES, AttackType


@dataclass
class ChannelConfig:
    n_qubits_per_round: int = 64
    benign_error_mean: float = 0.02
    benign_error_std: float = 0.008
    benign_error_clip: tuple[float, float] = (0.0, 0.06)
    seed: int | None = None


@dataclass
class EveState:
    """What the adversary is doing to this channel right now."""

    active: bool = False
    attack_type: AttackType = AttackType.BENIGN
    # Sampled once when an episode starts so the episode has one coherent
    # "personality", mirroring how the training streams were generated.
    channel_error_prob: float = 0.0
    eve_intercept_prob: float = 0.0
    intensity: float = 0.5
    started_at: float | None = None
    label: str = "benign"

    def as_dict(self) -> dict:
        return {
            "active": self.active,
            "attack_type": ATTACK_TYPE_NAMES[self.attack_type],
            "eve_intercept_prob": round(self.eve_intercept_prob, 4),
            "channel_error_prob": round(self.channel_error_prob, 4),
            "intensity": round(self.intensity, 3),
            "elapsed_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
            "label": self.label,
        }


@dataclass
class QberSample:
    t: int
    qber: float
    sifted_key_length: int
    n_intercepted: int
    n_qubits: int
    ground_truth_attack: bool
    ground_truth_type: str
    sim_ms: float
    wall_clock: float = field(default_factory=time.time)


class LiveQKDChannel:
    """Continuously executes BB84 rounds for one device."""

    def __init__(self, config: ChannelConfig | None = None):
        self.config = config or ChannelConfig()
        seed = self.config.seed
        self._rng = np.random.default_rng(seed)
        self.eve = EveState()
        self.t = 0
        self._lock = threading.Lock()

    # --- adversary control ------------------------------------------------
    def set_attack(
        self,
        attack_type: AttackType | None,
        intensity: float = 0.5,
        label: str = "",
    ) -> EveState:
        """Start or stop an attack episode on this channel.

        `intensity` in [0, 1] interpolates within the attack type's
        parameter range from `ATTACK_PROFILES`, so the same control means
        "how hard is Eve pushing" regardless of which attack is selected.
        """
        with self._lock:
            if attack_type is None or attack_type == AttackType.BENIGN:
                self.eve = EveState()
                return self.eve

            profile = ATTACK_PROFILES[attack_type]
            intensity = float(np.clip(intensity, 0.0, 1.0))
            chan_lo, chan_hi = profile["channel_error_prob_range"]
            eve_lo, eve_hi = profile["eve_intercept_prob_range"]

            self.eve = EveState(
                active=True,
                attack_type=attack_type,
                channel_error_prob=chan_lo + intensity * (chan_hi - chan_lo),
                eve_intercept_prob=eve_lo + intensity * (eve_hi - eve_lo),
                intensity=intensity,
                started_at=time.time(),
                label=label or ATTACK_TYPE_NAMES[attack_type],
            )
            return self.eve

    def clear_attack(self) -> EveState:
        return self.set_attack(None)

    # --- physics ----------------------------------------------------------
    def step(self) -> QberSample:
        """Execute one real BB84 round and return its QBER."""
        with self._lock:
            eve = self.eve

        if eve.active:
            channel_error = eve.channel_error_prob
            intercept = eve.eve_intercept_prob
        else:
            channel_error = float(
                np.clip(
                    self._rng.normal(
                        self.config.benign_error_mean, self.config.benign_error_std
                    ),
                    *self.config.benign_error_clip,
                )
            )
            intercept = 0.0

        t0 = time.perf_counter()
        result = simulate_bb84_round(
            n_qubits=self.config.n_qubits_per_round,
            channel_error_prob=channel_error,
            eve_intercept_prob=intercept,
            rng=self._rng,
        )
        sim_ms = (time.perf_counter() - t0) * 1000

        sample = QberSample(
            t=self.t,
            qber=result.qber,
            sifted_key_length=result.sifted_key_length,
            n_intercepted=result.n_intercepted,
            n_qubits=result.n_qubits,
            ground_truth_attack=eve.active,
            ground_truth_type=ATTACK_TYPE_NAMES[eve.attack_type],
            sim_ms=sim_ms,
        )
        self.t += 1
        return sample
