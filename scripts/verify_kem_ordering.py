"""
Build-time sanity check: the baseline KEM profile must be cheaper than the
hardened one.

The entire adaptive-gating claim rests on this ordering. If it ever inverts --
a mis-set reference table, a broken liboqs build, a swapped mechanism name --
then "escalate only when there is evidence" becomes strictly more expensive
than never escalating, and every CPU-saving figure the system reports flips
sign. That is a failure worth catching in CI and at image build time rather
than on a projector.

    python scripts/verify_kem_ordering.py

Exits non-zero if the ordering is wrong.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crypto_agility.kem_backend import (  # noqa: E402
    KEMProfile,
    get_kem_backend,
    is_liboqs_available,
)

MIN_RATIO = 2.0
N_SAMPLES = 5


def handshake_ms(backend, profile: KEMProfile) -> float:
    public_key, secret_handle, kg = backend.keygen(profile)
    ciphertext, ss_a, enc = backend.encapsulate(profile, public_key)
    ss_b, dec = backend.decapsulate(profile, secret_handle, ciphertext)
    if ss_a != ss_b:
        raise SystemExit(f"FAIL: {profile.value} shared secrets disagree")
    return kg.latency_ms + enc.latency_ms + dec.latency_ms


def main() -> int:
    backend = get_kem_backend()
    real = type(backend).__name__ == "LiboqsKEMBackend"
    print(f"KEM backend: {type(backend).__name__} (liboqs available: {is_liboqs_available()})")

    costs = {}
    for profile in (KEMProfile.BIKE_L1, KEMProfile.HQC_128):
        samples = [handshake_ms(backend, profile) for _ in range(N_SAMPLES)]
        costs[profile.value] = statistics.median(samples)
        print(f"  {profile.value:<10} {costs[profile.value]:7.3f} ms / handshake")

    baseline, hardened = costs["BIKE-L1"], costs["HQC-128"]
    ratio = hardened / max(baseline, 1e-9)
    print(f"  ratio (hardened / baseline): {ratio:.1f}x")

    if baseline >= hardened:
        print(
            "\nFAIL: the baseline profile is not cheaper than the hardened one.\n"
            "      Adaptive gating would report a NEGATIVE CPU saving.\n"
            "      Check _REFERENCE_PROFILE in crypto_agility/kem_backend.py,\n"
            "      or the liboqs mechanism mapping."
        )
        return 1

    if ratio < MIN_RATIO:
        print(
            f"\nWARNING: the profiles are within {ratio:.1f}x of each other. "
            f"Adaptive gating still saves, but far less than the reported figures."
        )

    print(f"\nOK — baseline is {ratio:.1f}x cheaper than hardened"
          f"{'' if real else ' (simulated backend)'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
