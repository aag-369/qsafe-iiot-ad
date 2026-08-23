"""
Per-link accounting for the demo: what the adaptive policy actually cost,
and what an always-on hardened profile would have cost over the same traffic.

The paper's headline 73.2% figure is a batch benchmark over a fixed 2,000
round stream. On a live link the equivalent number has to be accumulated as
it happens, which is what `LinkMetrics` does: every real handshake is added
to `adaptive_kem_ms`, and the same event is priced against a
static-always-HQC-128 counterfactual so the console can show a running
"CPU saved" counter that is derived from measurements, not asserted.

Two figures are tracked, because they answer two different questions and
conflating them would misrepresent the result:

`cpu_saved_pct` -- **session cost.** What this link actually spent on
rekeys, against a counterfactual charged one hardened handshake per rekey
the adaptive policy actually performed. A production link rekeys on profile
change, not per packet, so this number is small in absolute terms and
deliberately conservative: it does not invent handshakes the adaptive policy
avoided.

`paper_equivalent_saved_pct` -- **per-round cost.** The methodology of
`orchestrator/pipeline.py` and of the published benchmark: one KEM handshake
per round at the active profile, against one HQC-128 handshake per round.
This is the figure comparable to the paper's 73.2%, and the one to quote
alongside it.

Both are derived from per-profile handshake costs measured on the host at
startup, not from a table of assumed constants.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class LinkMetrics:
    """Running totals for one protected link."""

    started_at: float = field(default_factory=time.time)

    frames_sent: int = 0
    frames_received: int = 0
    frames_rejected: int = 0
    plaintext_bytes: int = 0
    ciphertext_bytes: int = 0

    n_handshakes: int = 0
    adaptive_kem_ms: float = 0.0
    static_hqc_kem_ms: float = 0.0

    # Paper-equivalent per-round accounting (see class docstring).
    round_adaptive_ms: float = 0.0
    round_static_ms: float = 0.0

    # Pre-seeded with both profiles rather than left to grow on first use.
    # These dicts are written from the control-loop worker and iterated from
    # the web layer; a *new key* appearing mid-iteration raises
    # "dictionary changed size during iteration", and the first moment a new
    # key would ever appear is the first escalation -- the single instant of
    # the demo that most needs to not throw.
    rounds_by_profile: dict[str, int] = field(
        default_factory=lambda: defaultdict(int, {"BIKE-L1": 0, "HQC-128": 0})
    )
    handshake_ms_by_profile: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list, {"BIKE-L1": [], "HQC-128": []})
    )

    # Reference cost of one handshake at each profile, measured on this host
    # at startup rather than assumed, so the counterfactual is honest about
    # the machine it is running on.
    hqc_reference_ms: float = 0.0
    bike_reference_ms: float = 0.0

    _recent_latency: deque = field(default_factory=lambda: deque(maxlen=64))

    def record_handshake(self, profile: str, total_ms: float) -> None:
        self.n_handshakes += 1
        self.adaptive_kem_ms += total_ms
        self.handshake_ms_by_profile[profile].append(total_ms)
        self._recent_latency.append(total_ms)
        if profile == "HQC-128":
            # The static baseline would have performed *this exact*
            # handshake, so charge it what it actually cost. Charging a
            # reference figure measured on an idle process at startup makes
            # the counterfactual cheaper than reality under load, which can
            # drive the reported saving negative.
            self.static_hqc_kem_ms += total_ms
        else:
            self.static_hqc_kem_ms += self.hqc_reference_ms or total_ms

    def record_round(self, profile: str) -> None:
        """Count one QKD round, and price it both ways.

        `round_adaptive_ms` charges this round the reference cost of one
        handshake at the profile that was actually active; `round_static_ms`
        charges it a hardened handshake. This mirrors
        `orchestrator/pipeline.py`, which performs exactly one handshake per
        round in both the adaptive and the static-baseline scenario, and is
        what makes the live "CPU saved" figure comparable to the 73.2%
        reported in the paper.
        """
        self.rounds_by_profile[profile] += 1
        ref = self.bike_reference_ms if profile == "BIKE-L1" else self.hqc_reference_ms
        self.round_adaptive_ms += ref
        self.round_static_ms += self.hqc_reference_ms

    def record_sent(self, plaintext_len: int, ciphertext_len: int) -> None:
        self.frames_sent += 1
        self.plaintext_bytes += plaintext_len
        self.ciphertext_bytes += ciphertext_len

    def record_received(self) -> None:
        self.frames_received += 1

    def record_rejected(self) -> None:
        self.frames_rejected += 1

    @property
    def paper_equivalent_saved_pct(self) -> float:
        """Per-round CPU saving under the paper's methodology.

        This is the number directly comparable to the published 73.2%: it
        prices one KEM handshake per round at the active profile against one
        HQC-128 handshake per round. It converges toward the published figure
        as the live attack duty cycle approaches the test stream's.
        """
        if self.round_static_ms <= 0:
            return 0.0
        return max(0.0, (1 - self.round_adaptive_ms / self.round_static_ms) * 100)

    @property
    def baseline_round_fraction(self) -> float:
        """Fraction of rounds spent on the low-overhead baseline profile."""
        total = sum(self.rounds_by_profile.values())
        if not total:
            return 0.0
        return self.rounds_by_profile.get("BIKE-L1", 0) / total

    @property
    def cpu_saved_pct(self) -> float:
        if self.static_hqc_kem_ms <= 0:
            return 0.0
        # Clamped for the same reason cpu_saved_ms is: the two must never
        # disagree about whether anything was saved.
        return max(0.0, (1 - self.adaptive_kem_ms / self.static_hqc_kem_ms) * 100)

    @property
    def cpu_saved_ms(self) -> float:
        return max(0.0, self.static_hqc_kem_ms - self.adaptive_kem_ms)

    @property
    def aead_overhead_bytes(self) -> int:
        """Total expansion the AEAD added over the plaintext it carried."""
        return self.ciphertext_bytes - self.plaintext_bytes

    @property
    def median_handshake_ms(self) -> float:
        if not self._recent_latency:
            return 0.0
        return statistics.median(self._recent_latency)

    def snapshot(self) -> dict:
        # Iterate over a copy: the worker thread may append to these while
        # the web layer is serializing them.
        handshakes = {k: list(v) for k, v in list(self.handshake_ms_by_profile.items())}
        by_profile = {
            profile: {
                "handshakes": len(vals),
                "median_ms": round(statistics.median(vals), 3) if vals else 0.0,
                "total_ms": round(sum(vals), 3),
            }
            for profile, vals in handshakes.items()
        }
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "frames_sent": self.frames_sent,
            "frames_received": self.frames_received,
            "frames_rejected": self.frames_rejected,
            "plaintext_bytes": self.plaintext_bytes,
            "ciphertext_bytes": self.ciphertext_bytes,
            "aead_overhead_bytes": self.aead_overhead_bytes,
            "n_handshakes": self.n_handshakes,
            "adaptive_kem_ms": round(self.adaptive_kem_ms, 3),
            "static_hqc_kem_ms": round(self.static_hqc_kem_ms, 3),
            "cpu_saved_ms": round(self.cpu_saved_ms, 3),
            "cpu_saved_pct": round(self.cpu_saved_pct, 2),
            "round_adaptive_ms": round(self.round_adaptive_ms, 3),
            "round_static_ms": round(self.round_static_ms, 3),
            "paper_equivalent_saved_pct": round(self.paper_equivalent_saved_pct, 2),
            "baseline_round_fraction": round(self.baseline_round_fraction, 4),
            "median_handshake_ms": round(self.median_handshake_ms, 3),
            "rounds_by_profile": dict(list(self.rounds_by_profile.items())),
            "by_profile": by_profile,
            "hqc_reference_ms": round(self.hqc_reference_ms, 3),
            "bike_reference_ms": round(self.bike_reference_ms, 3),
        }
