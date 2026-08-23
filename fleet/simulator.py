"""Runs several simulated IIoT devices side by side, each through the exact
same real pipeline used everywhere else in this project (BB84 -> binary GRU
detector -> hysteresis switch controller -> real liboqs KEM ops), then
attaches the additive attack-type classifier per device and hands all of it
to FleetCorrelator.

Three scenarios, chosen to make the fleet-level value visible:
  * "calm"                — no attacks anywhere. Nothing should ever alert.
  * "independent_attacks" — each device gets its own small, randomly-timed
    attack window (different types, different times) — realistic background
    noise. No coordination, so the correlator should NOT raise a fleet
    alert even though individual devices do escalate.
  * "coordinated_campaign" — a subset of devices are hit by the *same*
    attack type in the *same* time window (one adversary sweeping the
    fleet), while the rest of the fleet gets ordinary independent noise (or
    nothing). The correlator should isolate exactly the coordinated subset.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crypto_agility.kem_backend import KEMBackend
from crypto_agility.switch_controller import SwitchController
from fleet.correlator import FleetCorrelator, FleetCorrelatorConfig, FleetCorrelatorResult
from orchestrator.detector_runner import DetectorRunner
from orchestrator.pipeline import logs_to_dataframe, run_adaptive
from orchestrator.type_runner import AttackTypeRunner
from qkd_sim.qber_stream_multiclass import (
    ATTACKING_TYPES,
    AttackType,
    MultiClassQBERStreamGenerator,
    MultiClassStreamConfig,
)

VALID_SCENARIOS = ("calm", "independent_attacks", "coordinated_campaign")


@dataclass
class FleetConfig:
    n_devices: int = 6
    n_rounds: int = 80
    n_qubits_per_round: int = 48
    scenario: str = "coordinated_campaign"
    campaign_attack_type: AttackType = AttackType.EAVESDROP
    campaign_fraction: float = 0.5  # fraction of the fleet hit by the coordinated campaign
    campaign_window_len: int = 24
    escalate_threshold: float = 0.5
    de_escalate_threshold: float | None = None
    cooldown_rounds: int = 4
    seed: int | None = None


@dataclass
class DeviceResult:
    device_id: str
    df: pd.DataFrame  # raw stream: t, qber, label, attack_type (ground truth)
    confidence: np.ndarray  # binary detector confidence per round
    predicted_type: np.ndarray  # argmax attack-type index per round
    pipeline_df: pd.DataFrame  # profile/escalation/KEM-latency per round
    is_campaign_target: bool


@dataclass
class FleetResult:
    config: FleetConfig
    devices: list[DeviceResult] = field(default_factory=list)
    correlator_result: FleetCorrelatorResult | None = None
    wall_clock_s: float = 0.0


class FleetSimulator:
    def __init__(
        self,
        detector_runner: DetectorRunner,
        type_runner: AttackTypeRunner,
        kem_backend: KEMBackend,
        correlator: FleetCorrelator | None = None,
    ):
        self.detector_runner = detector_runner
        self.type_runner = type_runner
        self.kem_backend = kem_backend
        self.correlator = correlator or FleetCorrelator()

    def run(self, config: FleetConfig) -> FleetResult:
        if config.scenario not in VALID_SCENARIOS:
            raise ValueError(f"scenario must be one of {VALID_SCENARIOS}, got {config.scenario!r}")

        t0 = time.time()
        seed = config.seed if config.seed is not None else int(time.time() * 1000) % (2**31)
        rng = np.random.default_rng(seed)

        device_ids = [f"device-{i + 1:02d}" for i in range(config.n_devices)]

        campaign_targets: set[str] = set()
        campaign_start = 0
        if config.scenario == "coordinated_campaign":
            n_hit = max(1, round(config.campaign_fraction * config.n_devices))
            n_hit = min(n_hit, config.n_devices)
            campaign_targets = set(
                rng.choice(device_ids, size=n_hit, replace=False).tolist()
            )
            latest_start = max(1, config.n_rounds - config.campaign_window_len - 5)
            campaign_start = int(rng.integers(low=min(5, latest_start), high=max(6, latest_start + 1)))

        de_escalate = (
            config.de_escalate_threshold
            if config.de_escalate_threshold is not None
            else max(0.05, config.escalate_threshold - 0.36)
        )

        devices: list[DeviceResult] = []
        for dev_id in device_ids:
            dev_seed = int(rng.integers(0, 2**31 - 1))
            is_target = dev_id in campaign_targets

            if config.scenario == "calm":
                stream_cfg = MultiClassStreamConfig(
                    n_rounds=config.n_rounds,
                    n_qubits_per_round=config.n_qubits_per_round,
                    n_attack_windows=0,
                    seed=dev_seed,
                )
            elif config.scenario == "independent_attacks":
                # Deliberately sparse — one short, randomly-timed, randomly-
                # typed window per device. Dense enough that most devices
                # show *some* individual activity over the run, sparse
                # enough that 3+ devices rarely land on the same few rounds
                # by chance, so a fleet alert here would be a real (rare)
                # coincidence rather than the norm — which is exactly the
                # contrast this scenario exists to demonstrate.
                stream_cfg = MultiClassStreamConfig(
                    n_rounds=config.n_rounds,
                    n_qubits_per_round=config.n_qubits_per_round,
                    n_attack_windows=1,
                    attack_len_range=(5, max(6, config.n_rounds // 10)),
                    seed=dev_seed,
                )
            else:  # coordinated_campaign
                if is_target:
                    forced = [(campaign_start, config.campaign_window_len, config.campaign_attack_type)]
                    stream_cfg = MultiClassStreamConfig(
                        n_rounds=config.n_rounds,
                        n_qubits_per_round=config.n_qubits_per_round,
                        forced_windows=forced,
                        seed=dev_seed,
                    )
                else:
                    # Rest of the fleet: mostly calm, with a small chance of
                    # unrelated, incidental noise — so the demo shows the
                    # correlator correctly ignoring background static while
                    # catching the real campaign.
                    n_windows = 1 if rng.random() < 0.5 else 0
                    stream_cfg = MultiClassStreamConfig(
                        n_rounds=config.n_rounds,
                        n_qubits_per_round=config.n_qubits_per_round,
                        n_attack_windows=n_windows,
                        attack_len_range=(5, max(6, config.n_rounds // 8)),
                        seed=dev_seed,
                    )

            gen = MultiClassQBERStreamGenerator(stream_cfg)
            stream = gen.generate(verbose=False)
            df = stream.df

            confidence = self.detector_runner.score_stream(df)
            type_probs = self.type_runner.score_stream(df)
            predicted_type = np.argmax(type_probs, axis=1)

            controller = SwitchController(
                escalate_threshold=config.escalate_threshold,
                de_escalate_threshold=de_escalate,
                cooldown_rounds=config.cooldown_rounds,
            )
            logs = run_adaptive(df, confidence, self.kem_backend, controller)
            pipeline_df = logs_to_dataframe(logs)

            devices.append(
                DeviceResult(
                    device_id=dev_id,
                    df=df,
                    confidence=confidence,
                    predicted_type=predicted_type,
                    pipeline_df=pipeline_df,
                    is_campaign_target=is_target,
                )
            )

        device_confidences = {d.device_id: d.confidence for d in devices}
        device_types = {d.device_id: d.predicted_type for d in devices}
        correlator_result = self.correlator.analyze(device_confidences, device_types)

        return FleetResult(
            config=config,
            devices=devices,
            correlator_result=correlator_result,
            wall_clock_s=time.time() - t0,
        )
