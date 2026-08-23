"""Core per-round pipeline shared by both the adaptive and the static
baseline benchmark runs: given a QBER stream and a detector confidence
source, drive the crypto-agile switch and perform one real KEM handshake
(keygen + encapsulate + decapsulate) per round at whatever profile is
currently active."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crypto_agility.kem_backend import KEMBackend, KEMProfile
from crypto_agility.switch_controller import SwitchController


@dataclass
class RoundLog:
    t: int
    label: int
    confidence: float
    profile: str
    keygen_ms: float
    encaps_ms: float
    decaps_ms: float
    total_ms: float
    escalated: bool
    de_escalated: bool


def run_static_profile(
    df: pd.DataFrame,
    backend: KEMBackend,
    profile: KEMProfile,
) -> list[RoundLog]:
    """Baseline scenario: the given profile is active for every round,
    regardless of detector output (e.g. "always-on HQC-128")."""
    logs = []
    for row in df.itertuples():
        pk, sk_handle, kg = backend.keygen(profile)
        ct, ss1, enc = backend.encapsulate(profile, pk)
        ss2, dec = backend.decapsulate(profile, sk_handle, ct)
        total = kg.latency_ms + enc.latency_ms + dec.latency_ms
        logs.append(
            RoundLog(
                t=int(row.t),
                label=int(row.label),
                confidence=float("nan"),
                profile=profile.value,
                keygen_ms=kg.latency_ms,
                encaps_ms=enc.latency_ms,
                decaps_ms=dec.latency_ms,
                total_ms=total,
                escalated=False,
                de_escalated=False,
            )
        )
    return logs


def run_adaptive(
    df: pd.DataFrame,
    confidences: np.ndarray,
    backend: KEMBackend,
    controller: SwitchController,
) -> list[RoundLog]:
    """AI-gated scenario: the crypto profile is whatever the switch
    controller currently holds, driven by the GRU detector's per-round
    confidence score."""
    assert len(confidences) == len(df), "confidences must align 1:1 with df rows"
    logs = []
    for row, conf in zip(df.itertuples(), confidences):
        decision = controller.step(int(row.t), float(conf))
        profile = decision.profile

        pk, sk_handle, kg = backend.keygen(profile)
        ct, ss1, enc = backend.encapsulate(profile, pk)
        ss2, dec = backend.decapsulate(profile, sk_handle, ct)
        total = kg.latency_ms + enc.latency_ms + dec.latency_ms

        logs.append(
            RoundLog(
                t=int(row.t),
                label=int(row.label),
                confidence=float(conf),
                profile=profile.value,
                keygen_ms=kg.latency_ms,
                encaps_ms=enc.latency_ms,
                decaps_ms=dec.latency_ms,
                total_ms=total,
                escalated=decision.escalated_this_round,
                de_escalated=decision.de_escalated_this_round,
            )
        )
    return logs


def logs_to_dataframe(logs: list[RoundLog]) -> pd.DataFrame:
    return pd.DataFrame([vars(log) for log in logs])
