import numpy as np

from fleet.correlator import FleetCorrelator, FleetCorrelatorConfig
from qkd_sim.qber_stream_multiclass import AttackType


def _zeros(n_devices, n_rounds, device_prefix="dev"):
    conf = {f"{device_prefix}-{i}": np.zeros(n_rounds) for i in range(n_devices)}
    types = {f"{device_prefix}-{i}": np.zeros(n_rounds, dtype=int) for i in range(n_devices)}
    return conf, types


def test_calm_fleet_raises_no_alerts():
    conf, types = _zeros(6, 50)
    result = FleetCorrelator().analyze(conf, types)
    assert result.alerts == []
    assert not result.fleet_elevated_mask.any()


def test_coordinated_same_type_escalation_raises_one_alert():
    conf, types = _zeros(6, 50)
    for d in ["dev-0", "dev-1", "dev-2"]:
        conf[d][20:30] = 0.9
        types[d][20:30] = int(AttackType.EAVESDROP)

    result = FleetCorrelator(FleetCorrelatorConfig(min_devices=3, confidence_threshold=0.5)).analyze(conf, types)
    assert len(result.alerts) == 1
    alert = result.alerts[0]
    assert set(alert.device_ids) == {"dev-0", "dev-1", "dev-2"}
    assert alert.dominant_attack_type == "eavesdrop"
    assert alert.type_agreement == 1.0
    assert alert.t_start == 20 and alert.t_end == 29


def test_isolated_single_device_does_not_raise_fleet_alert():
    conf, types = _zeros(6, 50)
    conf["dev-0"][10:20] = 0.95
    types["dev-0"][10:20] = int(AttackType.JAMMING)

    result = FleetCorrelator(FleetCorrelatorConfig(min_devices=3)).analyze(conf, types)
    assert result.alerts == []


def test_simultaneous_but_different_attack_types_do_not_correlate():
    # Three devices escalate at the same time, but with three DIFFERENT
    # attack types — this is the key discriminator: simultaneous timing
    # alone isn't enough, they must also agree on what's happening.
    conf, types = _zeros(6, 50)
    conf["dev-0"][10:20] = 0.9
    types["dev-0"][10:20] = int(AttackType.EAVESDROP)
    conf["dev-1"][10:20] = 0.9
    types["dev-1"][10:20] = int(AttackType.JAMMING)
    conf["dev-2"][10:20] = 0.9
    types["dev-2"][10:20] = int(AttackType.PNS)

    result = FleetCorrelator(FleetCorrelatorConfig(min_devices=3)).analyze(conf, types)
    assert result.alerts == []


def test_below_confidence_threshold_does_not_count_as_elevated():
    conf, types = _zeros(6, 50)
    for d in ["dev-0", "dev-1", "dev-2"]:
        conf[d][20:30] = 0.4  # below default 0.5 threshold
        types[d][20:30] = int(AttackType.EAVESDROP)

    result = FleetCorrelator(FleetCorrelatorConfig(min_devices=3, confidence_threshold=0.5)).analyze(conf, types)
    assert result.alerts == []


def test_brief_gap_merges_into_one_episode():
    conf, types = _zeros(6, 60)
    for d in ["dev-0", "dev-1", "dev-2"]:
        conf[d][10:20] = 0.9
        types[d][10:20] = int(AttackType.JAMMING)
        # a 2-round dip (within merge_gap_rounds=3) then resumes
        conf[d][22:30] = 0.9
        types[d][22:30] = int(AttackType.JAMMING)

    result = FleetCorrelator(FleetCorrelatorConfig(min_devices=3, merge_gap_rounds=3)).analyze(conf, types)
    assert len(result.alerts) == 1
    assert result.alerts[0].t_start == 10
    assert result.alerts[0].t_end == 29


def test_empty_fleet_returns_no_alerts():
    result = FleetCorrelator().analyze({}, {})
    assert result.alerts == []
