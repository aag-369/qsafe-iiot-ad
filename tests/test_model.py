import numpy as np

from ai_detector.model import build_gru_model, count_params


def test_model_forward_pass_shape():
    model = build_gru_model(window_size=20, n_features=3, gru_units=8, dense_units=4)
    x = np.random.default_rng(0).normal(size=(5, 20, 3)).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (5, 1)
    assert np.all((y >= 0) & (y <= 1))  # sigmoid output


def test_model_param_count_reasonable_for_cortex_m4():
    model = build_gru_model(window_size=20, n_features=3, gru_units=32, dense_units=16)
    n_params = count_params(model)
    # A few thousand params is comfortably within Cortex-M4 flash/RAM
    # budgets after INT8 quantization; guard against accidental bloat.
    assert 0 < n_params < 20_000


def test_unroll_true_matches_unroll_false_outputs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 20, 3)).astype(np.float32)

    m1 = build_gru_model(window_size=20, n_features=3, gru_units=8, dense_units=4, unroll=False)
    m2 = build_gru_model(window_size=20, n_features=3, gru_units=8, dense_units=4, unroll=True)
    m2.set_weights(m1.get_weights())

    y1 = m1.predict(x, verbose=0)
    y2 = m2.predict(x, verbose=0)
    assert np.allclose(y1, y2, atol=1e-5)
