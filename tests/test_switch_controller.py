from crypto_agility.kem_backend import KEMProfile
from crypto_agility.switch_controller import SwitchController


def test_starts_on_baseline():
    sc = SwitchController()
    assert sc.current_profile is KEMProfile.BIKE_L1


def test_escalates_immediately_on_high_confidence():
    sc = SwitchController(escalate_threshold=0.5)
    d = sc.step(0, 0.9)
    assert d.profile is KEMProfile.HQC_128
    assert d.escalated_this_round


def test_does_not_deescalate_before_cooldown_elapses():
    sc = SwitchController(escalate_threshold=0.5, de_escalate_threshold=0.2, cooldown_rounds=5)
    sc.step(0, 0.9)  # escalate
    for t in range(1, 4):
        d = sc.step(t, 0.1)
        assert d.profile is KEMProfile.HQC_128
        assert not d.de_escalated_this_round


def test_deescalates_after_cooldown_elapses():
    sc = SwitchController(escalate_threshold=0.5, de_escalate_threshold=0.2, cooldown_rounds=3)
    sc.step(0, 0.9)  # escalate
    sc.step(1, 0.1)
    sc.step(2, 0.1)
    d = sc.step(3, 0.1)  # 3rd consecutive quiet round -> de-escalate
    assert d.profile is KEMProfile.BIKE_L1
    assert d.de_escalated_this_round


def test_dead_band_holds_current_profile_without_advancing_cooldown():
    sc = SwitchController(escalate_threshold=0.8, de_escalate_threshold=0.3, cooldown_rounds=2)
    sc.step(0, 0.9)  # escalate to HQC-128
    d = sc.step(1, 0.5)  # in the dead band [0.3, 0.8) -> hold, reset streak
    assert d.profile is KEMProfile.HQC_128
    d = sc.step(2, 0.1)  # 1st quiet round
    d = sc.step(3, 0.1)  # 2nd quiet round -> de-escalate
    assert d.profile is KEMProfile.BIKE_L1


def test_invalid_thresholds_raise():
    import pytest

    with pytest.raises(ValueError):
        SwitchController(escalate_threshold=0.3, de_escalate_threshold=0.5)
