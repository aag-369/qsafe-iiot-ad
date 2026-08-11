"""Lightweight GRU-based QBER anomaly detector, quantization-optimized for
ARM Cortex-M4 class microcontrollers."""

from .features import WindowConfig, build_windows
from .model import build_gru_model

__all__ = ["WindowConfig", "build_windows", "build_gru_model"]
