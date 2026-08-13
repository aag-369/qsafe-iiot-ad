"""Fleet-level correlation: takes each device's per-round confidence (from
the existing binary GRU detector) and per-round attack-type prediction
(from the additive attack-type classifier) and decides whether elevated
readings across multiple devices represent a *coordinated campaign* rather
than isolated, unrelated per-device noise or independent attacks.

The core idea, in plain terms: one device escalating is just that device
doing its job. Several devices escalating *at the same time*, especially
with the *same* attack-type tag, is much stronger evidence of a single
adversary sweeping across the fleet — exactly the kind of signal a
single-device view cannot see on its own.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from qkd_sim.qber_stream_multiclass import ATTACK_TYPE_NAMES, AttackType


@dataclass
class FleetAlert:
    t_start: int
    t_end: int
    device_ids: list[str]
    peak_device_count: int
    dominant_attack_type: str
    type_agreement: float  # fraction of (device, round) type-votes in the
    # episode that matched the dominant type, among non-benign votes


@dataclass
class FleetCorrelatorConfig:
    # A round counts as "fleet-elevated" if at least this many devices have
    # confidence >= confidence_threshold at that round.
    min_devices: int = 3
    confidence_threshold: float = 0.5
    # Consecutive fleet-elevated rounds separated by a gap no larger than
    # this many calm rounds are merged into a single episode, so a brief
    # dip below threshold mid-attack doesn't split one campaign into two
    # alerts.
    merge_gap_rounds: int = 3


@dataclass
class FleetCorrelatorResult:
    alerts: list[FleetAlert] = field(default_factory=list)
    fleet_elevated_mask: np.ndarray | None = None  # bool array, len = n_rounds
    elevated_device_counts: np.ndarray | None = None  # int array, len = n_rounds


class FleetCorrelator:
    def __init__(self, config: FleetCorrelatorConfig | None = None):
        self.config = config or FleetCorrelatorConfig()

    def analyze(
        self,
        device_confidences: dict[str, np.ndarray],
        device_types: dict[str, np.ndarray],
    ) -> FleetCorrelatorResult:
        """`device_confidences` and `device_types` are dicts of device_id ->
        1D arrays (same length across all devices): per-round binary-detector
        confidence, and per-round argmax attack-type index, respectively.

        A round only counts as "fleet-elevated" if enough devices are BOTH
        individually escalated AND agree on the same attack type at that
        round — simultaneous-but-unrelated escalations (different devices,
        different incidental attack types, coincidentally overlapping in
        time) are exactly the false-positive case this is designed to
        reject, since two unrelated attackers hitting different devices with
        different techniques at the same moment is a coincidence, not a
        campaign. Requiring type agreement is what tells them apart.
        """
        cfg = self.config
        device_ids = list(device_confidences.keys())
        if not device_ids:
            return FleetCorrelatorResult(alerts=[], fleet_elevated_mask=np.array([]), elevated_device_counts=np.array([]))

        n_rounds = len(next(iter(device_confidences.values())))
        elevated_counts = np.zeros(n_rounds, dtype=int)
        elevated_by_round: list[list[str]] = [[] for _ in range(n_rounds)]

        for t in range(n_rounds):
            # Devices elevated at this round, grouped by their predicted
            # attack type (benign-tagged elevations can't form a group).
            by_type: dict[int, list[str]] = {}
            for dev_id in device_ids:
                if device_confidences[dev_id][t] >= cfg.confidence_threshold:
                    tv = int(device_types[dev_id][t])
                    if tv != int(AttackType.BENIGN):
                        by_type.setdefault(tv, []).append(dev_id)

            if by_type:
                best_type, best_group = max(by_type.items(), key=lambda kv: len(kv[1]))
                elevated_counts[t] = len(best_group)
                elevated_by_round[t] = best_group

        fleet_elevated = elevated_counts >= cfg.min_devices

        # Group consecutive (allowing small gaps) fleet-elevated rounds into
        # episodes.
        episodes: list[tuple[int, int]] = []
        t = 0
        while t < n_rounds:
            if not fleet_elevated[t]:
                t += 1
                continue
            start = t
            end = t
            gap = 0
            t += 1
            while t < n_rounds:
                if fleet_elevated[t]:
                    end = t
                    gap = 0
                else:
                    gap += 1
                    if gap > cfg.merge_gap_rounds:
                        break
                t += 1
            episodes.append((start, end))

        alerts = []
        for start, end in episodes:
            involved = sorted(set(d for row in elevated_by_round[start : end + 1] for d in row))
            peak = max(elevated_counts[start : end + 1])

            # Majority attack type among non-benign (device, round) votes
            # from the *involved* devices within the episode window.
            votes = Counter()
            for dev_id in involved:
                types_arr = device_types[dev_id][start : end + 1]
                for tv in types_arr:
                    if tv != AttackType.BENIGN:
                        votes[int(tv)] += 1

            if votes:
                dominant_type_idx, dominant_count = votes.most_common(1)[0]
                total_votes = sum(votes.values())
                agreement = dominant_count / total_votes
                dominant_name = ATTACK_TYPE_NAMES[AttackType(dominant_type_idx)]
            else:
                agreement = 0.0
                dominant_name = "unknown"

            alerts.append(
                FleetAlert(
                    t_start=start,
                    t_end=end,
                    device_ids=involved,
                    peak_device_count=peak,
                    dominant_attack_type=dominant_name,
                    type_agreement=agreement,
                )
            )

        return FleetCorrelatorResult(
            alerts=alerts,
            fleet_elevated_mask=fleet_elevated,
            elevated_device_counts=elevated_counts,
        )
