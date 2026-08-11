import numpy as np
import pandas as pd

from crypto_agility.kem_backend import KEMProfile, get_kem_backend
from crypto_agility.switch_controller import SwitchController
from orchestrator.pipeline import logs_to_dataframe, run_adaptive, run_static_profile


def _make_df(n=20):
    return pd.DataFrame({"t": np.arange(n), "qber": np.random.default_rng(0).uniform(0, 0.05, n), "label": [0] * n})


def test_static_profile_run_uses_only_that_profile():
    df = _make_df(10)
    backend = get_kem_backend(force_simulated=True)
    logs = run_static_profile(df, backend, KEMProfile.HQC_128)
    out = logs_to_dataframe(logs)
    assert len(out) == 10
    assert (out["profile"] == KEMProfile.HQC_128.value).all()
    assert (out["total_ms"] > 0).all()


def test_adaptive_run_switches_profile_with_confidence():
    df = _make_df(6)
    backend = get_kem_backend(force_simulated=True)
    controller = SwitchController(escalate_threshold=0.5, de_escalate_threshold=0.2, cooldown_rounds=2)
    confidences = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
    logs = run_adaptive(df, confidences, backend, controller)
    out = logs_to_dataframe(logs)

    assert out.loc[0, "profile"] == KEMProfile.BIKE_L1.value
    assert out.loc[2, "profile"] == KEMProfile.HQC_128.value
    assert out.loc[2, "escalated"]


def test_adaptive_requires_aligned_confidences():
    df = _make_df(5)
    backend = get_kem_backend(force_simulated=True)
    controller = SwitchController()
    try:
        run_adaptive(df, np.array([0.1, 0.2]), backend, controller)
        assert False, "expected AssertionError for misaligned confidences"
    except AssertionError:
        pass
