"""Crypto-agile KEM layer: switches between a low-overhead baseline profile
(BIKE-L1) and a hardened profile (HQC-128) based on the AI detector's
confidence output."""

from .kem_backend import KEMProfile, get_kem_backend
from .switch_controller import SwitchController, SwitchDecision

__all__ = ["KEMProfile", "get_kem_backend", "SwitchController", "SwitchDecision"]
