"""Tests for the protected-link layer (secure_channel/).

These run against the simulated KEM backend by default so they stay fast in
CI, but the properties under test -- key separation across profiles and
epochs, replay rejection, tamper rejection, direction binding -- are
backend-independent.
"""

from __future__ import annotations

import pytest

from crypto_agility.kem_backend import KEMProfile, get_kem_backend
from secure_channel.aead import AeadChannel, AeadError, AeadKeys, build_nonce
from secure_channel.session import QSafeSession, derive_traffic_keys, new_session_id


@pytest.fixture
def backend():
    return get_kem_backend(force_simulated=True)


@pytest.fixture
def session(backend):
    return QSafeSession("test-session", backend=backend)


# --- key schedule ---------------------------------------------------------
def test_traffic_keys_are_32_bytes_and_distinct():
    keys = derive_traffic_keys(b"\x01" * 32, b"sid", 0, KEMProfile.BIKE_L1)
    assert len(keys.i2r_key) == 32
    assert len(keys.r2i_key) == 32
    assert keys.i2r_key != keys.r2i_key


def test_same_shared_secret_yields_different_keys_per_profile():
    """A BIKE-L1 epoch and an HQC-128 epoch must never share traffic keys,
    even in the pathological case of an identical KEM shared secret."""
    ss = b"\x02" * 32
    bike = derive_traffic_keys(ss, b"sid", 0, KEMProfile.BIKE_L1)
    hqc = derive_traffic_keys(ss, b"sid", 0, KEMProfile.HQC_128)
    assert bike.i2r_key != hqc.i2r_key
    assert bike.r2i_key != hqc.r2i_key


def test_same_shared_secret_yields_different_keys_per_epoch():
    ss = b"\x03" * 32
    e0 = derive_traffic_keys(ss, b"sid", 0, KEMProfile.BIKE_L1)
    e1 = derive_traffic_keys(ss, b"sid", 1, KEMProfile.BIKE_L1)
    assert e0.i2r_key != e1.i2r_key


def test_keys_are_session_scoped():
    ss = b"\x04" * 32
    a = derive_traffic_keys(ss, b"session-a", 0, KEMProfile.BIKE_L1)
    b = derive_traffic_keys(ss, b"session-b", 0, KEMProfile.BIKE_L1)
    assert a.i2r_key != b.i2r_key


def test_nonce_is_96_bits_and_unique_per_seq():
    assert len(build_nonce(0, 0)) == 12
    assert build_nonce(0, 1) != build_nonce(0, 0)
    assert build_nonce(1, 0) != build_nonce(0, 0)


def test_aead_keys_reject_wrong_length():
    with pytest.raises(ValueError):
        AeadKeys(epoch=0, i2r_key=b"short", r2i_key=b"\x00" * 32)


# --- round trip -----------------------------------------------------------
def test_uplink_round_trip(session):
    seq, ct = session.seal_uplink(b"pressure=4.2")
    assert session.open_uplink(seq, ct) == b"pressure=4.2"


def test_downlink_round_trip(session):
    seq, ct = session.seal_downlink(b"valve=closed")
    assert session.open_downlink(seq, ct) == b"valve=closed"


def test_ciphertext_expands_by_gcm_tag(session):
    plaintext = b"x" * 100
    _, ct = session.seal_uplink(plaintext)
    assert len(ct) == len(plaintext) + 16  # AES-GCM 128-bit tag


def test_many_frames_in_order(session):
    for i in range(200):
        seq, ct = session.seal_uplink(f"frame-{i}".encode())
        assert session.open_uplink(seq, ct) == f"frame-{i}".encode()


# --- security properties --------------------------------------------------
def test_replay_is_rejected(session):
    seq, ct = session.seal_uplink(b"open valve")
    session.open_uplink(seq, ct)
    with pytest.raises(AeadError, match="replayed"):
        session.open_uplink(seq, ct)


def test_tampered_ciphertext_is_rejected(session):
    seq, ct = session.seal_uplink(b"setpoint=10")
    tampered = bytes([ct[0] ^ 0xFF]) + ct[1:]
    with pytest.raises(AeadError, match="authentication failed"):
        session.open_uplink(seq, tampered)


def test_frame_cannot_be_replayed_in_the_opposite_direction(session):
    """Direction is bound into the AAD, so an uplink frame replayed as a
    downlink frame must fail even though both directions exist in the same
    epoch."""
    seq, ct = session.seal_uplink(b"telemetry")
    with pytest.raises(AeadError):
        session.open_downlink(seq, ct)


def test_frame_from_previous_epoch_fails_after_rekey(session):
    seq, ct = session.seal_uplink(b"pre-rekey frame")
    session.rekey(KEMProfile.HQC_128, reason="escalate")
    with pytest.raises(AeadError):
        session.open_uplink(seq, ct)


def test_forged_future_frame_does_not_advance_replay_window(session):
    """A forged frame with a high sequence number must not be able to shift
    the replay window forward and lock out legitimate traffic."""
    real_seq, real_ct = session.seal_uplink(b"legit")
    forged = b"\x00" * 32
    with pytest.raises(AeadError):
        session.open_uplink(9999, forged)
    assert session.open_uplink(real_seq, real_ct) == b"legit"


def test_out_of_window_frame_is_rejected(session):
    keys = derive_traffic_keys(b"\x05" * 32, b"sid", 0, KEMProfile.BIKE_L1)
    tx = AeadChannel(keys=keys, is_initiator=True)
    rx = AeadChannel(keys=keys, is_initiator=False)
    frames = [tx.seal(f"f{i}".encode()) for i in range(200)]
    for seq, ct in frames[100:]:
        rx.open(seq, ct)
    with pytest.raises(AeadError, match="older than the replay window"):
        seq, ct = frames[0]
        rx.open(seq, ct)


def test_out_of_order_within_window_is_accepted(session):
    frames = [session.seal_uplink(f"f{i}".encode()) for i in range(8)]
    for seq, ct in reversed(frames):
        assert session.open_uplink(seq, ct) == f"f{seq}".encode()


# --- rekey behaviour ------------------------------------------------------
def test_rekey_increments_epoch_and_changes_fingerprint(session):
    before = session.key_fingerprint
    epoch_before = session.epoch
    session.rekey(KEMProfile.HQC_128, reason="escalate")
    assert session.epoch == epoch_before + 1
    assert session.key_fingerprint != before


def test_rekey_records_the_mechanism_actually_used(session):
    rec = session.rekey(KEMProfile.HQC_128, reason="escalate")
    assert rec.profile == "HQC-128"
    assert "HQC" in rec.kem_mechanism
    assert rec.total_ms > 0


def test_traffic_works_after_rekey(session):
    session.rekey(KEMProfile.HQC_128, reason="escalate")
    seq, ct = session.seal_uplink(b"post-rekey")
    assert session.open_uplink(seq, ct) == b"post-rekey"
    session.rekey(KEMProfile.BIKE_L1, reason="de-escalate")
    seq, ct = session.seal_uplink(b"post-de-escalate")
    assert session.open_uplink(seq, ct) == b"post-de-escalate"


def test_sequence_numbers_restart_each_epoch(session):
    seq_a, _ = session.seal_uplink(b"a")
    session.rekey(KEMProfile.HQC_128)
    seq_b, _ = session.seal_uplink(b"b")
    assert seq_a == 0 and seq_b == 0


def test_session_state_reports_simulated_backend_honestly(backend):
    s = QSafeSession("labelled", backend=backend)
    assert s.state()["simulated_kem"] is True


def test_new_session_id_is_unique():
    assert new_session_id() != new_session_id()
