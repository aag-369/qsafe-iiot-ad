"""Exercises the trained attack-type classifier end to end (loads the real
committed model artifact, same as how orchestrator/detector_runner.py's
binary counterpart is only exercised indirectly via the web API tests)."""

import os

import numpy as np
import pandas as pd
import pytest

MODEL_PATH = "models/attack_type_gru.keras"
NORM_PATH = "models/attack_type_norm_stats.json"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(MODEL_PATH) and os.path.exists(NORM_PATH)),
    reason="trained attack-type model artifacts not present",
)


def _make_stream(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "t": np.arange(n),
            "qber": rng.uniform(0.0, 0.05, n),
            "label": [0] * n,
            "attack_type": [0] * n,
        }
    )


def test_score_stream_shape_and_probability_simplex():
    from orchestrator.type_runner import AttackTypeRunner

    runner = AttackTypeRunner(model_path=MODEL_PATH, norm_stats_path=NORM_PATH, window_size=20)
    df = _make_stream(40)
    probs = runner.score_stream(df)
    assert probs.shape == (40, 4)
    # every row should sum close to 1 (softmax simplex), including the
    # zero-padded warm-up rows (which default to [1,0,0,0])
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_predict_types_returns_valid_class_indices():
    from orchestrator.type_runner import AttackTypeRunner

    runner = AttackTypeRunner(model_path=MODEL_PATH, norm_stats_path=NORM_PATH, window_size=20)
    df = _make_stream(30)
    types = runner.predict_types(df)
    assert types.shape == (30,)
    assert set(np.unique(types)) <= {0, 1, 2, 3}


def test_warmup_rounds_default_to_benign():
    from orchestrator.type_runner import AttackTypeRunner

    runner = AttackTypeRunner(model_path=MODEL_PATH, norm_stats_path=NORM_PATH, window_size=20)
    df = _make_stream(25)
    probs = runner.score_stream(df)
    # first (window_size - 1) rounds have no full window yet
    assert np.allclose(probs[:19, 0], 1.0)
