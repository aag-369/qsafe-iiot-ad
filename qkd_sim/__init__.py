"""Physical-layer QKD simulation package (BB84 via Qiskit)."""

from .bb84 import simulate_bb84_round
from .qber_stream import QBERStreamGenerator, StreamConfig

__all__ = ["simulate_bb84_round", "QBERStreamGenerator", "StreamConfig"]
