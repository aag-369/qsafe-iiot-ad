import pytest

from crypto_agility.kem_backend import KEMProfile, get_kem_backend, is_liboqs_available


@pytest.mark.parametrize("profile", [KEMProfile.BIKE_L1, KEMProfile.HQC_128])
def test_simulated_backend_roundtrip(profile):
    backend = get_kem_backend(force_simulated=True)
    pk, sk, kg = backend.keygen(profile)
    ct, ss1, enc = backend.encapsulate(profile, pk)
    ss2, dec = backend.decapsulate(profile, sk, ct)

    assert kg.simulated and enc.simulated and dec.simulated
    assert len(ss1) == len(ss2) > 0
    assert kg.public_key_bytes == len(pk)
    assert enc.ciphertext_bytes == len(ct)


@pytest.mark.skipif(not is_liboqs_available(), reason="liboqs C library not built in this environment")
@pytest.mark.parametrize("profile", [KEMProfile.BIKE_L1, KEMProfile.HQC_128])
def test_real_liboqs_backend_roundtrip(profile):
    backend = get_kem_backend()
    pk, sk_handle, kg = backend.keygen(profile)
    ct, ss1, enc = backend.encapsulate(profile, pk)
    ss2, dec = backend.decapsulate(profile, sk_handle, ct)

    assert not kg.simulated
    # The defining correctness property of a KEM: both parties derive the
    # same shared secret.
    assert ss1 == ss2
    assert len(ss1) > 0
