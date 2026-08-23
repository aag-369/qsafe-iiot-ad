"""Q-Safe Field Link: a live, device-to-device demonstration of the
Q-Safe IIoT-AD framework.

Every layer of the paper's Fig. 1 runs here in real time against real
devices, using the committed model artifacts and real liboqs KEM operations
where available. Nothing in `qkd_sim/`, `ai_detector/`, `crypto_agility/`,
`orchestrator/` or `fleet/` is modified by this package -- it imports them.
"""

from .channel import ChannelConfig, EveState, LiveQKDChannel
from .detector import LiveDetector, LiveTypeTagger
from .node import EdgeNode, NodeConfig
from .runtime import LinkRuntime
from .scenarios import SCENARIOS, apply_scenario

__all__ = [
    "ChannelConfig",
    "EveState",
    "LiveQKDChannel",
    "LiveDetector",
    "LiveTypeTagger",
    "EdgeNode",
    "NodeConfig",
    "LinkRuntime",
    "SCENARIOS",
    "apply_scenario",
]
