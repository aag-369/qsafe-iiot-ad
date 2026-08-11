"""
KEM abstraction over liboqs (Open Quantum Safe), providing the two NIST
Round-4 Key Encapsulation Mechanism profiles used by the crypto-agile
switch:

  * BIKE_L1  — "lattice-lite" / code-based, low-overhead baseline profile.
  * HQC_128  — hardened, IND-CCA2, constant-time-resilient profile.

If liboqs is available at import time (real `oqs` bindings + built
`liboqs.so`), all operations run real PQC keypair generation, encapsulation
and decapsulation. If it is not available (e.g. a CI runner without the
liboqs C library built), the module falls back to a *simulated* backend that
reproduces the correct wire-format sizes and per-operation timing profile
from the published liboqs benchmark corpus, so the rest of the pipeline
(orchestrator, benchmark harness) runs unmodified either way. Every
measurement this fallback returns is tagged `simulated=True` so it can never
be silently mistaken for a hardware measurement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

try:
    import oqs  # type: ignore

    _OQS_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when liboqs isn't built
    _OQS_AVAILABLE = False


class KEMProfile(Enum):
    BIKE_L1 = "BIKE-L1"
    HQC_128 = "HQC-128"


# Maps our profile names to the exact liboqs mechanism identifiers. liboqs
# names HQC's Category-1 (128-bit security) parameter set "HQC-1".
_LIBOQS_MECHANISM = {
    KEMProfile.BIKE_L1: "BIKE-L1",
    KEMProfile.HQC_128: "HQC-1",
}

# Published reference sizes/timings (liboqs benchmark corpus, x86_64
# reference build) used only by the simulated fallback so it stays
# representative when the real library can't be built in a given
# environment.
_REFERENCE_PROFILE = {
    KEMProfile.BIKE_L1: {
        "public_key_bytes": 1541,
        "secret_key_bytes": 5223,
        "ciphertext_bytes": 1573,
        "shared_secret_bytes": 32,
        "keygen_ms": 0.35,
        "encaps_ms": 0.10,
        "decaps_ms": 1.40,
    },
    KEMProfile.HQC_128: {
        "public_key_bytes": 2249,
        "secret_key_bytes": 2305,
        "ciphertext_bytes": 4481,
        "shared_secret_bytes": 32,
        "keygen_ms": 0.09,
        "encaps_ms": 0.16,
        "decaps_ms": 0.22,
    },
}


@dataclass
class KEMOperationResult:
    profile: KEMProfile
    op: str  # "keygen" | "encaps" | "decaps"
    latency_ms: float
    public_key_bytes: int = 0
    ciphertext_bytes: int = 0
    shared_secret_bytes: int = 0
    simulated: bool = False


class KEMBackend:
    """Common interface implemented by both the real (liboqs) and simulated
    backends, so callers never need to branch on availability."""

    def keygen(self, profile: KEMProfile) -> tuple[bytes, object, KEMOperationResult]:
        raise NotImplementedError

    def encapsulate(self, profile: KEMProfile, public_key: bytes) -> tuple[bytes, bytes, KEMOperationResult]:
        raise NotImplementedError

    def decapsulate(
        self, profile: KEMProfile, secret_key_handle: object, ciphertext: bytes
    ) -> tuple[bytes, KEMOperationResult]:
        raise NotImplementedError


class LiboqsKEMBackend(KEMBackend):
    """Real PQC operations via Open Quantum Safe's liboqs."""

    def keygen(self, profile: KEMProfile):
        mech = _LIBOQS_MECHANISM[profile]
        kem = oqs.KeyEncapsulation(mech)
        t0 = time.perf_counter()
        public_key = kem.generate_keypair()
        latency_ms = (time.perf_counter() - t0) * 1000
        result = KEMOperationResult(
            profile=profile,
            op="keygen",
            latency_ms=latency_ms,
            public_key_bytes=len(public_key),
            simulated=False,
        )
        # `kem` itself holds the secret key internally; we return it as the
        # opaque "secret key handle" the caller must keep alive to decaps.
        return public_key, kem, result

    def encapsulate(self, profile: KEMProfile, public_key: bytes):
        mech = _LIBOQS_MECHANISM[profile]
        with oqs.KeyEncapsulation(mech) as kem:
            t0 = time.perf_counter()
            ciphertext, shared_secret = kem.encap_secret(public_key)
            latency_ms = (time.perf_counter() - t0) * 1000
        result = KEMOperationResult(
            profile=profile,
            op="encaps",
            latency_ms=latency_ms,
            ciphertext_bytes=len(ciphertext),
            shared_secret_bytes=len(shared_secret),
            simulated=False,
        )
        return ciphertext, shared_secret, result

    def decapsulate(self, profile: KEMProfile, secret_key_handle: "oqs.KeyEncapsulation", ciphertext: bytes):
        t0 = time.perf_counter()
        shared_secret = secret_key_handle.decap_secret(ciphertext)
        latency_ms = (time.perf_counter() - t0) * 1000
        secret_key_handle.free()
        result = KEMOperationResult(
            profile=profile,
            op="decaps",
            latency_ms=latency_ms,
            shared_secret_bytes=len(shared_secret),
            simulated=False,
        )
        return shared_secret, result


class SimulatedKEMBackend(KEMBackend):
    """Deterministic stand-in used when liboqs isn't built in the current
    environment. Produces correctly-sized dummy artifacts and sleeps for the
    reference-benchmark latency so downstream CPU/latency accounting stays
    representative. Never used silently: `KEMOperationResult.simulated` is
    always True."""

    def keygen(self, profile: KEMProfile):
        ref = _REFERENCE_PROFILE[profile]
        t0 = time.perf_counter()
        time.sleep(ref["keygen_ms"] / 1000)
        latency_ms = (time.perf_counter() - t0) * 1000
        public_key = bytes(ref["public_key_bytes"])
        secret_key = bytes(ref["secret_key_bytes"])
        result = KEMOperationResult(
            profile=profile,
            op="keygen",
            latency_ms=latency_ms,
            public_key_bytes=len(public_key),
            simulated=True,
        )
        return public_key, secret_key, result

    def encapsulate(self, profile: KEMProfile, public_key: bytes):
        ref = _REFERENCE_PROFILE[profile]
        t0 = time.perf_counter()
        time.sleep(ref["encaps_ms"] / 1000)
        latency_ms = (time.perf_counter() - t0) * 1000
        ciphertext = bytes(ref["ciphertext_bytes"])
        shared_secret = bytes(ref["shared_secret_bytes"])
        result = KEMOperationResult(
            profile=profile,
            op="encaps",
            latency_ms=latency_ms,
            ciphertext_bytes=len(ciphertext),
            shared_secret_bytes=len(shared_secret),
            simulated=True,
        )
        return ciphertext, shared_secret, result

    def decapsulate(self, profile: KEMProfile, secret_key_handle: bytes, ciphertext: bytes):
        ref = _REFERENCE_PROFILE[profile]
        t0 = time.perf_counter()
        time.sleep(ref["decaps_ms"] / 1000)
        latency_ms = (time.perf_counter() - t0) * 1000
        shared_secret = bytes(ref["shared_secret_bytes"])
        result = KEMOperationResult(
            profile=profile,
            op="decaps",
            latency_ms=latency_ms,
            shared_secret_bytes=len(shared_secret),
            simulated=True,
        )
        return shared_secret, result


def get_kem_backend(force_simulated: bool = False) -> KEMBackend:
    """Returns the real liboqs backend if available, otherwise the
    simulated fallback. Set `force_simulated=True` to explicitly use the
    fallback (useful for fast unit tests / CI without a liboqs build)."""
    if not force_simulated and _OQS_AVAILABLE:
        try:
            enabled = oqs.get_enabled_kem_mechanisms()
            required = set(_LIBOQS_MECHANISM.values())
            if required.issubset(set(enabled)):
                return LiboqsKEMBackend()
        except Exception:
            pass
    return SimulatedKEMBackend()


def is_liboqs_available() -> bool:
    return _OQS_AVAILABLE
