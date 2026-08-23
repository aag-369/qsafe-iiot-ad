import numpy as np

from qkd_sim.bb84 import simulate_bb84_round


def test_qber_in_valid_range():
    rng = np.random.default_rng(0)
    result = simulate_bb84_round(32, channel_error_prob=0.02, eve_intercept_prob=0.0, rng=rng)
    assert 0.0 <= result.qber <= 1.0
    assert result.n_qubits == 32
    assert 0 <= result.sifted_key_length <= 32


def test_intercept_resend_raises_qber_on_average():
    """Eve's intercept-resend attack should statistically raise QBER above
    the benign baseline (textbook BB84 result: ~25% error per intercepted,
    wrong-basis-resent qubit)."""
    rng = np.random.default_rng(1)
    benign = [
        simulate_bb84_round(64, channel_error_prob=0.01, eve_intercept_prob=0.0, rng=rng).qber
        for _ in range(8)
    ]
    attacked = [
        simulate_bb84_round(64, channel_error_prob=0.01, eve_intercept_prob=0.9, rng=rng).qber
        for _ in range(8)
    ]
    assert np.mean(attacked) > np.mean(benign)


def test_no_intercept_is_reproducible_with_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    r1 = simulate_bb84_round(32, channel_error_prob=0.0, eve_intercept_prob=0.0, rng=rng1)
    r2 = simulate_bb84_round(32, channel_error_prob=0.0, eve_intercept_prob=0.0, rng=rng2)
    assert r1.qber == r2.qber
    assert r1.sifted_key_length == r2.sifted_key_length
