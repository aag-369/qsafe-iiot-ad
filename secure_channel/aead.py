"""
Authenticated encryption for the Q-Safe protected link.

The existing `orchestrator/pipeline.py` performs a real KEM handshake every
round and measures its latency, but discards the shared secret -- correct
for a cost benchmark, but it means no application data is ever actually
protected. This module closes that gap: it turns a KEM shared secret into a
working AES-256-GCM channel so two real endpoints can exchange real frames.

Nonce construction
------------------
A 96-bit GCM nonce is built as::

    nonce = epoch (uint32, big-endian) || seq (uint64, big-endian)

`epoch` is incremented by the session layer on every rekey (i.e. every time
the crypto-agility switch flips profiles) and `seq` is a strictly
monotonic per-direction counter. Because the traffic keys are *also*
re-derived on every rekey, a nonce can never repeat under a given key even
if an epoch were somehow reused.

Directionality
--------------
Initiator->responder and responder->initiator traffic use *separate* keys
derived from the same shared secret (see session.py), so the two directions
can never collide in nonce space regardless of counter values.

Replay protection
-----------------
Each receiving direction keeps a 64-frame sliding bitmap window. Frames
older than the window, or already seen inside it, are rejected. This is the
same construction DTLS and IPsec use, and it matters for the demo: an
industrial link that accepts a replayed "valve closed" frame is not secure
just because the bytes were encrypted.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Frame direction identifiers. These select which traffic key is used and
# keep the two directions in disjoint nonce spaces.
DIR_INITIATOR_TO_RESPONDER = 0
DIR_RESPONDER_TO_INITIATOR = 1

_REPLAY_WINDOW_BITS = 64


class AeadError(Exception):
    """Raised when a frame fails to authenticate, replays, or is malformed."""


def build_nonce(epoch: int, seq: int) -> bytes:
    """96-bit deterministic nonce: epoch || seq."""
    if not 0 <= epoch < 2**32:
        raise ValueError(f"epoch out of range: {epoch}")
    if not 0 <= seq < 2**64:
        raise ValueError(f"seq out of range: {seq}")
    return struct.pack(">IQ", epoch, seq)


def key_fingerprint(key: bytes) -> str:
    """Short, non-reversible identifier for a traffic key.

    Displayed in the demo UIs so an observer can *see* the key change at the
    moment of a rekey, without the key itself ever being rendered.
    """
    return hashlib.sha256(b"qsafe-fp\x00" + key).hexdigest()[:16]


@dataclass
class _ReplayWindow:
    """Sliding-window replay detector over a monotonic sequence space."""

    highest: int = -1
    bitmap: int = 0

    def check_and_update(self, seq: int) -> None:
        if seq < 0:
            raise AeadError(f"negative sequence number {seq}")

        if seq > self.highest:
            shift = seq - self.highest
            if shift >= _REPLAY_WINDOW_BITS:
                self.bitmap = 0
            else:
                self.bitmap <<= shift
                self.bitmap &= (1 << _REPLAY_WINDOW_BITS) - 1
            self.bitmap |= 1
            self.highest = seq
            return

        offset = self.highest - seq
        if offset >= _REPLAY_WINDOW_BITS:
            raise AeadError(
                f"frame seq={seq} is older than the replay window "
                f"(highest={self.highest})"
            )
        mask = 1 << offset
        if self.bitmap & mask:
            raise AeadError(f"replayed frame seq={seq}")
        self.bitmap |= mask


@dataclass
class AeadKeys:
    """The two directional traffic keys for one epoch."""

    epoch: int
    i2r_key: bytes
    r2i_key: bytes

    def __post_init__(self) -> None:
        for name, k in (("i2r_key", self.i2r_key), ("r2i_key", self.r2i_key)):
            if len(k) != 32:
                raise ValueError(f"{name} must be 32 bytes (AES-256), got {len(k)}")

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the epoch as a whole (both directions combined)."""
        return key_fingerprint(self.i2r_key + self.r2i_key)


@dataclass
class AeadChannel:
    """One endpoint's view of the protected channel for a single epoch.

    `is_initiator` selects which key this endpoint seals with and which it
    opens with, so both endpoints construct an `AeadChannel` from identical
    `AeadKeys` and automatically end up mirrored.
    """

    keys: AeadKeys
    is_initiator: bool
    _send_seq: int = 0
    _replay: _ReplayWindow = field(default_factory=_ReplayWindow)

    @property
    def epoch(self) -> int:
        return self.keys.epoch

    @property
    def fingerprint(self) -> str:
        return self.keys.fingerprint

    @property
    def frames_sent(self) -> int:
        return self._send_seq

    @property
    def _send_key(self) -> bytes:
        return self.keys.i2r_key if self.is_initiator else self.keys.r2i_key

    @property
    def _recv_key(self) -> bytes:
        return self.keys.r2i_key if self.is_initiator else self.keys.i2r_key

    @property
    def _send_dir(self) -> int:
        return DIR_INITIATOR_TO_RESPONDER if self.is_initiator else DIR_RESPONDER_TO_INITIATOR

    def seal(self, plaintext: bytes, aad: bytes = b"") -> tuple[int, bytes]:
        """Encrypt one frame. Returns (seq, ciphertext_with_tag)."""
        seq = self._send_seq
        self._send_seq += 1
        nonce = build_nonce(self.epoch, seq)
        full_aad = self._frame_aad(self._send_dir, seq, aad)
        ct = AESGCM(self._send_key).encrypt(nonce, plaintext, full_aad)
        return seq, ct

    def open(self, seq: int, ciphertext: bytes, aad: bytes = b"") -> bytes:
        """Decrypt and authenticate one frame, enforcing replay protection.

        Replay state is only advanced once the tag verifies, so a forged
        frame carrying a future sequence number cannot be used to shift the
        window forward and lock out legitimate traffic.
        """
        recv_dir = (
            DIR_RESPONDER_TO_INITIATOR if self.is_initiator else DIR_INITIATOR_TO_RESPONDER
        )
        nonce = build_nonce(self.epoch, seq)
        full_aad = self._frame_aad(recv_dir, seq, aad)
        try:
            plaintext = AESGCM(self._recv_key).decrypt(nonce, ciphertext, full_aad)
        except InvalidTag as exc:
            raise AeadError(
                f"authentication failed for frame seq={seq} epoch={self.epoch}"
            ) from exc
        self._replay.check_and_update(seq)
        return plaintext

    def _frame_aad(self, direction: int, seq: int, extra: bytes) -> bytes:
        """Bind epoch, direction and sequence into the authenticated data.

        Without this an attacker could take a valid frame and replay it in
        the opposite direction or under a different epoch header; the tag
        would still verify because GCM authenticates only what it is given.
        """
        return struct.pack(">IBQ", self.epoch, direction, seq) + extra
