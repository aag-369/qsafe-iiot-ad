import numpy as np
import pandas as pd

from ai_detector.features import WindowConfig, build_windows, normalize_features


def _make_df(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "t": np.arange(n),
            "qber": rng.uniform(0, 0.1, n),
            "label": (rng.uniform(0, 1, n) > 0.8).astype(int),
        }
    )


def test_build_windows_shapes():
    df = _make_df(50)
    cfg = WindowConfig(window_size=10, stride=1)
    X, y = build_windows(df, cfg)
    assert X.shape == (50 - 10 + 1, 10, len(cfg.features))
    assert y.shape == (50 - 10 + 1,)
    assert set(np.unique(y)).issubset({0.0, 1.0})


def test_window_label_matches_final_timestep():
    df = _make_df(30)
    cfg = WindowConfig(window_size=5)
    X, y = build_windows(df, cfg)
    # First window covers indices [0:5), label should equal df.label[4].
    assert y[0] == df["label"].iloc[4]
    assert y[-1] == df["label"].iloc[-1]


def test_normalize_features_zero_mean_unit_std():
    X = np.random.default_rng(0).normal(5, 2, size=(100, 10, 3)).astype(np.float32)
    X_norm, mean, std = normalize_features(X)
    flat = X_norm.reshape(-1, 3)
    assert np.allclose(flat.mean(axis=0), 0, atol=1e-3)
    assert np.allclose(flat.std(axis=0), 1, atol=1e-3)

    # Reusing mean/std on new data should not recompute them.
    X2 = np.random.default_rng(1).normal(5, 2, size=(20, 10, 3)).astype(np.float32)
    X2_norm, mean2, std2 = normalize_features(X2, mean=mean, std=std)
    assert np.array_equal(mean, mean2)
    assert np.array_equal(std, std2)
