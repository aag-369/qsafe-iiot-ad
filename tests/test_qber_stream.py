from qkd_sim.qber_stream import QBERStreamGenerator, StreamConfig


def test_generate_stream_shape_and_labels():
    cfg = StreamConfig(
        n_rounds=60,
        n_qubits_per_round=16,
        n_attack_windows=2,
        attack_len_range=(5, 10),
        seed=1,
    )
    gen = QBERStreamGenerator(cfg)
    stream = gen.generate(verbose=False)
    df = stream.df

    assert len(df) == 60
    assert set(df.columns) >= {"t", "qber", "sifted_key_length", "n_intercepted", "label"}
    assert df["label"].isin([0, 1]).all()
    assert df["label"].sum() > 0  # at least some attack rounds were injected
    assert df["qber"].between(0.0, 1.0).all()


def test_attack_windows_do_not_overlap():
    cfg = StreamConfig(n_rounds=200, n_qubits_per_round=16, n_attack_windows=5, seed=2)
    gen = QBERStreamGenerator(cfg)
    import numpy as np

    rng = np.random.default_rng(cfg.seed)
    is_attack, intercept_prob = gen._sample_attack_windows(rng)
    # Count contiguous True runs; with non-overlap placement this should be
    # <= n_attack_windows (could be fewer if placement attempts ran out).
    runs = 0
    prev = False
    for v in is_attack:
        if v and not prev:
            runs += 1
        prev = v
    assert 0 < runs <= cfg.n_attack_windows
