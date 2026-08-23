import numpy as np

from ai_detector.model import build_gru_model, count_params


def test_binary_head_unaffected_by_n_classes_param():
    # n_classes=None must reproduce the exact original binary architecture —
    # this is the backward-compatibility guarantee the whole design leans
    # on: the core detector/switch-controller path is untouched.
    model = build_gru_model(window_size=20, n_features=3, gru_units=8, dense_units=4)
    x = np.random.default_rng(0).normal(size=(5, 20, 3)).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (5, 1)
    assert np.all((y >= 0) & (y <= 1))
    assert model.loss == "binary_crossentropy"


def test_multiclass_head_shape_and_softmax_output():
    model = build_gru_model(window_size=20, n_features=4, gru_units=8, dense_units=4, n_classes=4)
    x = np.random.default_rng(0).normal(size=(6, 20, 4)).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (6, 4)
    # softmax rows sum to 1
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5)
    assert np.all((y >= 0) & (y <= 1))


def test_multiclass_param_count_still_small():
    model = build_gru_model(window_size=20, n_features=4, gru_units=32, dense_units=16, n_classes=4)
    n_params = count_params(model)
    assert 0 < n_params < 20_000
