"""Protected-link layer: turns the crypto-agility layer's KEM profile
decision into a real AES-256-GCM session that carries application traffic,
rekeyed live whenever the profile changes."""

from .aead import AeadChannel, AeadError, AeadKeys, key_fingerprint
from .metrics import LinkMetrics
from .session import HandshakeRecord, QSafeSession, derive_traffic_keys, new_session_id

__all__ = [
    "AeadChannel",
    "AeadError",
    "AeadKeys",
    "key_fingerprint",
    "LinkMetrics",
    "HandshakeRecord",
    "QSafeSession",
    "derive_traffic_keys",
    "new_session_id",
]
