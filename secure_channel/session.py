"""
PQC session establishment for the Q-Safe protected link.

This is the piece that binds the crypto-agility layer to actual traffic.
`crypto_agility/` decides *which* KEM profile should be active;
`orchestrator/pipeline.py` measures what that profile costs. Neither of them
produces a usable session key. `QSafeSession` does:

    KEM (BIKE-L1 | HQC-128)  ->  shared secret  ->  HKDF-SHA256  ->  AES-256-GCM

and, crucially, re-runs that derivation *live* every time the switch
controller changes profile. A profile change in this system is therefore not
cosmetic: it is a real re-handshake under a different post-quantum mechanism,
producing genuinely different traffic keys, which the demo UIs surface as a
visible change in key fingerprint.

Handshake shape
---------------
A standard KEM-based key establishment, split into explicit steps so it can
run either in-process (both endpoints in the gateway) or across a real
transport:

    responder:  offer()            -> public key
    initiator:  respond(pk)        -> ciphertext, keys
    responder:  complete(ct)       -> keys

The responder holds the ephemeral decapsulation key; the initiator
encapsulates to it. Keys are ephemeral per epoch, so each rekey is
forward-secret with respect to earlier epochs.

Key schedule
------------
HKDF-SHA256 expands the KEM shared secret into 64 bytes, split into two
32-byte directional traffic keys. The `info` string binds the protocol
version, the session id, the epoch, and the KEM mechanism name, so keys
derived under BIKE-L1 can never collide with keys derived under HQC-128
even from an identical shared secret.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from crypto_agility.kem_backend import KEMBackend, KEMProfile, get_kem_backend

from .aead import AeadChannel, AeadKeys

PROTOCOL_LABEL = b"qsafe-iiot-ad/link/v1"


@dataclass
class HandshakeRecord:
    """Everything measurable about one rekey, for the console and the report."""

    epoch: int
    profile: str
    kem_mechanism: str
    public_key_bytes: int
    ciphertext_bytes: int
    shared_secret_bytes: int
    keygen_ms: float
    encaps_ms: float
    decaps_ms: float
    total_ms: float
    key_fingerprint: str
    simulated: bool
    reason: str = ""
    wall_clock: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "profile": self.profile,
            "kem_mechanism": self.kem_mechanism,
            "public_key_bytes": self.public_key_bytes,
            "ciphertext_bytes": self.ciphertext_bytes,
            "shared_secret_bytes": self.shared_secret_bytes,
            "keygen_ms": round(self.keygen_ms, 4),
            "encaps_ms": round(self.encaps_ms, 4),
            "decaps_ms": round(self.decaps_ms, 4),
            "total_ms": round(self.total_ms, 4),
            "key_fingerprint": self.key_fingerprint,
            "simulated": self.simulated,
            "reason": self.reason,
            "wall_clock": self.wall_clock,
        }


_MECHANISM_LABEL = {
    KEMProfile.BIKE_L1: "BIKE-L1",
    KEMProfile.HQC_128: "HQC-128 (liboqs HQC-1)",
}


def derive_traffic_keys(
    shared_secret: bytes,
    session_id: bytes,
    epoch: int,
    profile: KEMProfile,
) -> AeadKeys:
    """HKDF-SHA256 the KEM shared secret into two directional AES-256 keys."""
    info = b"|".join(
        [
            PROTOCOL_LABEL,
            session_id,
            str(epoch).encode(),
            profile.value.encode(),
        ]
    )
    okm = HKDF(
        algorithm=hashes.SHA256(),
        length=64,
        salt=session_id,
        info=info,
    ).derive(shared_secret)
    return AeadKeys(epoch=epoch, i2r_key=okm[:32], r2i_key=okm[32:])


class QSafeSession:
    """One protected link between an edge node and the gateway.

    Both endpoints are represented here because the demo gateway hosts both
    sides of the tunnel in-process by default; `rekey()` runs the full
    handshake and installs mirrored `AeadChannel`s. When the edge node runs
    as a separate process, the same three handshake primitives are driven
    over the wire instead (see `handshake_steps`).
    """

    def __init__(
        self,
        session_id: str,
        backend: KEMBackend | None = None,
        initial_profile: KEMProfile = KEMProfile.BIKE_L1,
    ):
        self.session_id = session_id
        self._sid_bytes = session_id.encode()
        self.backend = backend or get_kem_backend()
        self.epoch = -1
        self.profile: KEMProfile | None = None
        self.initiator: AeadChannel | None = None
        self.responder: AeadChannel | None = None
        self.handshakes: list[HandshakeRecord] = []
        # Rekeys are driven by the control-loop worker thread while frames
        # are sealed and opened from the web layer. Without this lock a
        # rekey landing between a seal and its matching open swaps the keys
        # underneath the frame, and the open fails authentication against a
        # frame that was perfectly valid when it was created.
        self._lock = threading.RLock()
        self.rekey(initial_profile, reason="session-open")

    # --- handshake ------------------------------------------------------
    def rekey(self, profile: KEMProfile, reason: str = "") -> HandshakeRecord:
        """Run a full KEM handshake under `profile` and install new keys.

        Called on session open and on every crypto-agility profile switch.
        The KEM work itself is done outside the lock -- an HQC-128 handshake
        takes ~9 ms and holding the lock for that long would stall traffic;
        only the key installation is serialized against in-flight frames.
        """
        epoch = self.epoch + 1

        public_key, secret_handle, kg = self.backend.keygen(profile)
        ciphertext, ss_initiator, enc = self.backend.encapsulate(profile, public_key)
        ss_responder, dec = self.backend.decapsulate(profile, secret_handle, ciphertext)

        if ss_initiator != ss_responder:
            raise RuntimeError(
                f"KEM shared secrets disagree for {profile.value} -- "
                "refusing to install traffic keys"
            )

        keys = derive_traffic_keys(ss_initiator, self._sid_bytes, epoch, profile)

        with self._lock:
            self.epoch = epoch
            self.profile = profile
            self.initiator = AeadChannel(keys=keys, is_initiator=True)
            self.responder = AeadChannel(keys=keys, is_initiator=False)

        record = HandshakeRecord(
            epoch=epoch,
            profile=profile.value,
            kem_mechanism=_MECHANISM_LABEL[profile],
            public_key_bytes=kg.public_key_bytes,
            ciphertext_bytes=enc.ciphertext_bytes,
            shared_secret_bytes=enc.shared_secret_bytes,
            keygen_ms=kg.latency_ms,
            encaps_ms=enc.latency_ms,
            decaps_ms=dec.latency_ms,
            total_ms=kg.latency_ms + enc.latency_ms + dec.latency_ms,
            key_fingerprint=keys.fingerprint,
            simulated=kg.simulated,
            reason=reason,
        )
        self.handshakes.append(record)
        return record

    # --- traffic --------------------------------------------------------
    def seal_uplink(self, plaintext: bytes, aad: bytes = b"") -> tuple[int, bytes]:
        """Edge node -> gateway."""
        with self._lock:
            assert self.initiator is not None
            return self.initiator.seal(plaintext, aad)

    def open_uplink(self, seq: int, ciphertext: bytes, aad: bytes = b"") -> bytes:
        with self._lock:
            assert self.responder is not None
            return self.responder.open(seq, ciphertext, aad)

    def seal_downlink(self, plaintext: bytes, aad: bytes = b"") -> tuple[int, bytes]:
        """Gateway -> edge node."""
        with self._lock:
            assert self.responder is not None
            return self.responder.seal(plaintext, aad)

    def open_downlink(self, seq: int, ciphertext: bytes, aad: bytes = b"") -> bytes:
        with self._lock:
            assert self.initiator is not None
            return self.initiator.open(seq, ciphertext, aad)

    def roundtrip_uplink(self, plaintext: bytes, aad: bytes = b"") -> tuple[int, bytes, bytes]:
        """Seal and immediately open one uplink frame under a single lock
        hold, so a concurrent rekey cannot land between the two halves."""
        with self._lock:
            seq, ciphertext = self.seal_uplink(plaintext, aad)
            opened = self.open_uplink(seq, ciphertext, aad)
            return seq, ciphertext, opened

    def roundtrip_downlink(self, plaintext: bytes, aad: bytes = b"") -> tuple[int, bytes, bytes]:
        with self._lock:
            seq, ciphertext = self.seal_downlink(plaintext, aad)
            opened = self.open_downlink(seq, ciphertext, aad)
            return seq, ciphertext, opened

    # --- introspection --------------------------------------------------
    @property
    def key_fingerprint(self) -> str:
        assert self.initiator is not None
        return self.initiator.fingerprint

    @property
    def last_handshake(self) -> HandshakeRecord:
        return self.handshakes[-1]

    @property
    def total_handshake_ms(self) -> float:
        return sum(h.total_ms for h in self.handshakes)

    def state(self) -> dict:
        h = self.last_handshake
        return {
            "session_id": self.session_id,
            "epoch": self.epoch,
            "profile": self.profile.value if self.profile else None,
            "kem_mechanism": h.kem_mechanism,
            "key_fingerprint": self.key_fingerprint,
            "simulated_kem": h.simulated,
            "n_rekeys": len(self.handshakes) - 1,
            "total_handshake_ms": round(self.total_handshake_ms, 3),
            "last_handshake_ms": round(h.total_ms, 3),
        }


def new_session_id(prefix: str = "sess") -> str:
    return f"{prefix}-{os.urandom(6).hex()}"


# --- explicit three-step handshake, for a node in a separate process ------
class ResponderHandshake:
    """Gateway side of a handshake driven over a real transport."""

    def __init__(self, backend: KEMBackend, profile: KEMProfile, session_id: str, epoch: int):
        self.backend = backend
        self.profile = profile
        self.session_id = session_id
        self.epoch = epoch
        self._secret_handle = None
        self._keygen = None

    def offer(self) -> bytes:
        public_key, self._secret_handle, self._keygen = self.backend.keygen(self.profile)
        return public_key

    def complete(self, ciphertext: bytes) -> tuple[AeadKeys, object, object]:
        if self._secret_handle is None:
            raise RuntimeError("offer() must be called before complete()")
        shared_secret, dec = self.backend.decapsulate(
            self.profile, self._secret_handle, ciphertext
        )
        keys = derive_traffic_keys(
            shared_secret, self.session_id.encode(), self.epoch, self.profile
        )
        return keys, self._keygen, dec


def initiator_respond(
    backend: KEMBackend,
    profile: KEMProfile,
    session_id: str,
    epoch: int,
    public_key: bytes,
) -> tuple[bytes, AeadKeys, object]:
    """Edge-node side: encapsulate to the gateway's offer and derive keys."""
    ciphertext, shared_secret, enc = backend.encapsulate(profile, public_key)
    keys = derive_traffic_keys(shared_secret, session_id.encode(), epoch, profile)
    return ciphertext, keys, enc
