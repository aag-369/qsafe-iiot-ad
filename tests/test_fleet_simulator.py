"""End-to-end fleet simulation test: small enough to run fast in the normal
pytest suite (force_simulated KEM backend, tiny device count/round count),
but exercises the real BB84 + trained models + switch controller + fleet
correlator all wired together."""

import os

import pytest

DETECTOR_MODEL = "models/gru_detector.keras"
DETECTOR_NORM = "models/norm_stats.json"
TYPE_MODEL = "models/attack_type_gru.keras"
TYPE_NORM = "models/attack_type_norm_stats.json"

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in [DETECTOR_MODEL, DETECTOR_NORM, TYPE_MODEL, TYPE_NORM]),
    reason="trained model artifacts not present",
)


def _build_simulator():
    from crypto_agility.kem_backend import get_kem_backend
    from fleet.correlator import FleetCorrelator, FleetCorrelatorConfig
    from fleet.simulator import FleetSimulator
    from orchestrator.detector_runner import DetectorRunner
    from orchestrator.type_runner import AttackTypeRunner

    detector = DetectorRunner(model_path=DETECTOR_MODEL, norm_stats_path=DETECTOR_NORM)
    type_runner = AttackTypeRunner(model_path=TYPE_MODEL, norm_stats_path=TYPE_NORM)
    backend = get_kem_backend(force_simulated=True)
    correlator = FleetCorrelator(FleetCorrelatorConfig(min_devices=2, confidence_threshold=0.5))
    return FleetSimulator(detector, type_runner, backend, correlator)


def test_calm_scenario_produces_devices_with_no_forced_attack_labels():
    from fleet.simulator import FleetConfig

    sim = _build_simulator()
    cfg = FleetConfig(n_devices=3, n_rounds=25, n_qubits_per_round=16, scenario="calm", seed=1)
    result = sim.run(cfg)

    assert len(result.devices) == 3
    for dev in result.devices:
        assert (dev.df["attack_type"] == 0).all()
        assert len(dev.pipeline_df) == 25


def test_coordinated_campaign_targets_are_recorded_and_forced_into_stream():
    from fleet.simulator import FleetConfig
    from qkd_sim.qber_stream_multiclass import AttackType

    sim = _build_simulator()
    cfg = FleetConfig(
        n_devices=4,
        n_rounds=40,
        n_qubits_per_round=16,
        scenario="coordinated_campaign",
        campaign_attack_type=AttackType.JAMMING,
        campaign_fraction=0.5,
        campaign_window_len=12,
        escalate_threshold=0.5,
        seed=42,
    )
    result = sim.run(cfg)

    targets = [d for d in result.devices if d.is_campaign_target]
    non_targets = [d for d in result.devices if not d.is_campaign_target]
    assert len(targets) == 2
    assert len(non_targets) == 2
    for dev in targets:
        assert (dev.df["attack_type"] == int(AttackType.JAMMING)).any()

    # correlator ran and returned a well-formed result either way (whether
    # or not it happened to clear the alert bar on this particular seed)
    assert result.correlator_result is not None
    assert result.correlator_result.elevated_device_counts is not None
    assert len(result.correlator_result.elevated_device_counts) == 40


def test_invalid_scenario_raises():
    from fleet.simulator import FleetConfig

    sim = _build_simulator()
    cfg = FleetConfig(n_devices=2, n_rounds=10, scenario="not-a-real-scenario")
    with pytest.raises(ValueError):
        sim.run(cfg)
