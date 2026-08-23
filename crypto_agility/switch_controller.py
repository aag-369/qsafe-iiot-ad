"""
Software-defined cryptographic switch: turns the GRU detector's per-round
confidence score into a KEM profile decision (BIKE-L1 baseline vs. HQC-128
hardened), with hysteresis so a single noisy score doesn't cause the
profile to flap back and forth every round.

Policy:
  - Escalate to HQC-128 the instant confidence >= escalate_threshold.
  - Only de-escalate back to BIKE-L1 after `cooldown_rounds` consecutive
    rounds with confidence < de_escalate_threshold (below the escalate
    threshold, i.e. a dead band, to avoid oscillation right at the
    boundary).
"""

from __future__ import annotations

from dataclasses import dataclass

from .kem_backend import KEMProfile


@dataclass
class SwitchDecision:
    t: int
    confidence: float
    profile: KEMProfile
    escalated_this_round: bool
    de_escalated_this_round: bool


class SwitchController:
    def __init__(
        self,
        escalate_threshold: float = 0.5,
        de_escalate_threshold: float = 0.3,
        cooldown_rounds: int = 10,
    ):
        if de_escalate_threshold > escalate_threshold:
            raise ValueError("de_escalate_threshold must be <= escalate_threshold")
        self.escalate_threshold = escalate_threshold
        self.de_escalate_threshold = de_escalate_threshold
        self.cooldown_rounds = cooldown_rounds

        self._current_profile = KEMProfile.BIKE_L1
        self._quiet_streak = 0

    @property
    def current_profile(self) -> KEMProfile:
        return self._current_profile

    def step(self, t: int, confidence: float) -> SwitchDecision:
        escalated = False
        de_escalated = False

        if confidence >= self.escalate_threshold:
            if self._current_profile is KEMProfile.BIKE_L1:
                escalated = True
            self._current_profile = KEMProfile.HQC_128
            self._quiet_streak = 0
        elif confidence < self.de_escalate_threshold:
            if self._current_profile is KEMProfile.HQC_128:
                self._quiet_streak += 1
                if self._quiet_streak >= self.cooldown_rounds:
                    self._current_profile = KEMProfile.BIKE_L1
                    de_escalated = True
                    self._quiet_streak = 0
        else:
            # In the dead band: hold current profile, don't advance the
            # quiet streak (avoids de-escalating on borderline scores).
            self._quiet_streak = 0

        return SwitchDecision(
            t=t,
            confidence=confidence,
            profile=self._current_profile,
            escalated_this_round=escalated,
            de_escalated_this_round=de_escalated,
        )
