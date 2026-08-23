import numpy as np

from qkd_sim.qber_stream_multiclass import (
    ATTACKING_TYPES,
    AttackType,
    MultiClassQBERStreamGenerator,
    MultiClassStreamConfig,
)


def test_calm_stream_is_all_benign():
    cfg = MultiClassStreamConfig(n_rounds=50, n_qubits_per_round=16, n_attack_windows=0, seed=1)
    df = MultiClassQBERStreamGenerator(cfg).generate(verbose=False).df
    assert (df["attack_type"] == int(AttackType.BENIGN)).all()
    assert (df["label"] == 0).all()


def test_random_windows_are_placed_and_labeled_consistently():
    cfg = MultiClassStreamConfig(n_rounds=200, n_qubits_per_round=16, n_attack_windows=5, seed=2)
    df = MultiClassQBERStreamGenerator(cfg).generate(verbose=False).df
    assert len(df) == 200
    # label (binary) must agree with attack_type (benign vs. not) on every round
    assert ((df["attack_type"] != int(AttackType.BENIGN)) == (df["label"] == 1)).all()
    # at least some attack rounds actually got placed
    assert (df["attack_type"] != int(AttackType.BENIGN)).sum() > 0
    assert set(df["attack_type"].unique()) <= {int(t) for t in AttackType}


def test_forced_window_places_exact_type_and_span():
    cfg = MultiClassStreamConfig(
        n_rounds=100,
        n_qubits_per_round=16,
        forced_windows=[(20, 15, AttackType.JAMMING)],
        seed=3,
    )
    df = MultiClassQBERStreamGenerator(cfg).generate(verbose=False).df
    in_window = df[(df["t"] >= 20) & (df["t"] < 35)]
    outside = df[(df["t"] < 20) | (df["t"] >= 35)]
    assert (in_window["attack_type"] == int(AttackType.JAMMING)).all()
    assert (outside["attack_type"] == int(AttackType.BENIGN)).all()


def test_jamming_produces_higher_qber_than_pns_on_average():
    # Sanity check on the attack-profile design: jamming (direct channel
    # noise injection) should read much louder than pns (deliberately
    # stealthy) on average, given the ATTACK_PROFILES parameter ranges.
    # Average over a full window (rather than one round) to smooth out
    # per-round stochasticity and keep the assertion non-flaky.
    cfg_jam = MultiClassStreamConfig(
        n_rounds=40, n_qubits_per_round=64,
        forced_windows=[(0, 40, AttackType.JAMMING)], seed=7,
    )
    jamming_qber = MultiClassQBERStreamGenerator(cfg_jam).generate(verbose=False).df["qber"].mean()

    cfg_pns = MultiClassStreamConfig(
        n_rounds=40, n_qubits_per_round=64,
        forced_windows=[(0, 40, AttackType.PNS)], seed=7,
    )
    pns_qber = MultiClassQBERStreamGenerator(cfg_pns).generate(verbose=False).df["qber"].mean()

    cfg_benign = MultiClassStreamConfig(n_rounds=40, n_qubits_per_round=64, n_attack_windows=0, seed=7)
    benign_qber = MultiClassQBERStreamGenerator(cfg_benign).generate(verbose=False).df["qber"].mean()

    assert jamming_qber > pns_qber > benign_qber


def test_attacking_types_excludes_benign():
    assert AttackType.BENIGN not in ATTACKING_TYPES
    assert len(ATTACKING_TYPES) == 3
