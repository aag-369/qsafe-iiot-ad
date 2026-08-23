"""
One IIoT edge node, wired end to end.

This is the per-node control loop from Fig. 1 of the paper, closed every
round against live telemetry instead of a pre-generated CSV:

    LiveQKDChannel  ->  LiveDetector  ->  SwitchController  ->  QSafeSession
      (BB84, q_t)       (GRU, c_t)        (hysteresis, P_t)     (rekey + AEAD)

The only component that is new relative to the published pipeline is the
last one. `orchestrator/pipeline.py` performs a KEM handshake per round and
discards the secret; here the handshake happens on *profile change* and the
resulting secret becomes the session key that protects real application
frames. That is both more realistic (production links do not re-handshake
every packet) and what makes a profile switch observable to a person
standing in front of the screen: the key fingerprint changes.

Thresholds, cooldown, window size and the detector itself are all loaded
from the committed artifacts, so the node on stage behaves exactly like the
node in the benchmark.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from crypto_agility.kem_backend import KEMBackend, KEMProfile
from crypto_agility.switch_controller import SwitchController
from qkd_sim.qber_stream_multiclass import AttackType
from secure_channel.metrics import LinkMetrics
from secure_channel.session import QSafeSession, new_session_id

from .channel import ChannelConfig, LiveQKDChannel, QberSample
from .detector import LiveDetector


@dataclass
class NodeConfig:
    device_id: str
    display_name: str = ""
    role: str = "field-sensor"
    n_qubits_per_round: int = 64
    rounds_per_second: float = 4.0
    escalate_threshold: float | None = None      # default: committed threshold
    de_escalate_threshold: float | None = None   # default: escalate - 0.36
    cooldown_rounds: int = 4
    history_len: int = 600
    seed: int | None = None


@dataclass
class RoundRecord:
    """One round of the control loop, as the console renders it."""

    t: int
    wall_clock: float
    qber: float
    confidence: float
    profile: str
    epoch: int
    escalated: bool
    de_escalated: bool
    ground_truth_attack: bool
    ground_truth_type: str
    predicted_type: str
    sim_ms: float
    handshake_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "t": self.t,
            "wall_clock": round(self.wall_clock, 3),
            "qber": round(self.qber, 5),
            "confidence": round(self.confidence, 5),
            "profile": self.profile,
            "epoch": self.epoch,
            "escalated": self.escalated,
            "de_escalated": self.de_escalated,
            "ground_truth_attack": self.ground_truth_attack,
            "ground_truth_type": self.ground_truth_type,
            "predicted_type": self.predicted_type,
            "sim_ms": round(self.sim_ms, 3),
            "handshake_ms": round(self.handshake_ms, 3),
        }


class EdgeNode:
    """A single device's full Q-Safe stack."""

    def __init__(
        self,
        config: NodeConfig,
        detector: LiveDetector,
        kem_backend: KEMBackend,
        hqc_reference_ms: float = 0.0,
        bike_reference_ms: float = 0.0,
    ):
        self.config = config
        self.device_id = config.device_id
        self.display_name = config.display_name or config.device_id
        self.detector = detector

        self.channel = LiveQKDChannel(
            ChannelConfig(n_qubits_per_round=config.n_qubits_per_round, seed=config.seed)
        )

        escalate = (
            config.escalate_threshold
            if config.escalate_threshold is not None
            else detector.threshold
        )
        de_escalate = (
            config.de_escalate_threshold
            if config.de_escalate_threshold is not None
            else max(0.05, escalate - 0.36)
        )
        self.controller = SwitchController(
            escalate_threshold=escalate,
            de_escalate_threshold=de_escalate,
            cooldown_rounds=config.cooldown_rounds,
        )
        self.escalate_threshold = escalate
        self.de_escalate_threshold = de_escalate

        self.session = QSafeSession(
            new_session_id(config.device_id), backend=kem_backend,
            initial_profile=KEMProfile.BIKE_L1,
        )
        self.metrics = LinkMetrics(
            hqc_reference_ms=hqc_reference_ms, bike_reference_ms=bike_reference_ms
        )
        self.metrics.record_handshake(
            self.session.last_handshake.profile, self.session.last_handshake.total_ms
        )

        # Monotonic counters. The handshake list published in state() is
        # capped for payload size, so counting escalations from it would make
        # the console's tally *decrease* once a long-running device passes
        # the cap.
        self.n_escalations = 0
        self.n_de_escalations = 0

        self.history: deque[RoundRecord] = deque(maxlen=config.history_len)
        self.qber_tail: deque[float] = deque(maxlen=max(64, detector.tail_len))
        self.predicted_type = "benign"
        self.predicted_type_confidence = 0.0
        self.last_payload: dict | None = None
        self.last_frame: dict | None = None
        self.connected_sensor = False
        self.connected_monitor = False
        self.created_at = time.time()
        self._lock = threading.Lock()

        # Per-node interval so nodes can tick at different rates if needed.
        self.tick_interval = 1.0 / max(0.5, config.rounds_per_second)
        self._next_tick = time.time()

    # --- control loop -------------------------------------------------------
    def due(self, now: float) -> bool:
        return now >= self._next_tick

    def step(self) -> RoundRecord:
        """Execute exactly one round of the per-node control loop."""
        # Schedule the next tick *before* doing the work. If anything below
        # raises -- Aer, TFLite, liboqs -- this node would otherwise stay
        # permanently "due" and the worker would retry it at the loop's full
        # rate, flooding the event deque and pushing real rounds out of it.
        self._next_tick = time.time() + self.tick_interval

        sample: QberSample = self.channel.step()
        # This node owns its QBER history; the detector is a shared,
        # stateless scorer (see LiveDetector.score_tail).
        self.qber_tail.append(sample.qber)
        confidence = self.detector.score_tail(self.qber_tail)
        decision = self.controller.step(sample.t, confidence)

        handshake_ms = 0.0
        if decision.escalated_this_round or decision.de_escalated_this_round:
            reason = "escalate" if decision.escalated_this_round else "de-escalate"
            record = self.session.rekey(decision.profile, reason=reason)
            handshake_ms = record.total_ms
            self.metrics.record_handshake(record.profile, record.total_ms)
            if decision.escalated_this_round:
                self.n_escalations += 1
            else:
                self.n_de_escalations += 1

        self.metrics.record_round(decision.profile.value)

        rec = RoundRecord(
            t=sample.t,
            wall_clock=sample.wall_clock,
            qber=sample.qber,
            confidence=confidence,
            profile=decision.profile.value,
            epoch=self.session.epoch,
            escalated=decision.escalated_this_round,
            de_escalated=decision.de_escalated_this_round,
            ground_truth_attack=sample.ground_truth_attack,
            ground_truth_type=sample.ground_truth_type,
            predicted_type=self.predicted_type,
            sim_ms=sample.sim_ms,
            handshake_ms=handshake_ms,
        )
        with self._lock:
            self.history.append(rec)
        return rec

    # --- application traffic ------------------------------------------------
    def send_uplink(self, payload: dict) -> dict:
        """Seal one application frame from the device to the gateway.

        Returns a description of what actually went on the wire, including a
        ciphertext preview -- the demo shows an observer's view of the link
        beside the authorized endpoint's view.
        """
        plaintext = json.dumps(payload, separators=(",", ":")).encode()
        # Seal, open, and read the epoch under one lock hold: a rekey landing
        # between the two halves would otherwise fail authentication on a
        # frame that was valid when it was created, and report the *next*
        # epoch's fingerprint back to the phone.
        with self.session._lock:
            seq, ciphertext, opened = self.session.roundtrip_uplink(plaintext)
            epoch = self.session.epoch
            profile = self.session.profile.value if self.session.profile else None
            fingerprint = self.session.key_fingerprint
        self.metrics.record_sent(len(plaintext), len(ciphertext))
        self.metrics.record_received()
        decoded = json.loads(opened)

        frame = {
            "device_id": self.device_id,
            "seq": seq,
            "epoch": epoch,
            "profile": profile,
            "key_fingerprint": fingerprint,
            "plaintext_bytes": len(plaintext),
            "ciphertext_bytes": len(ciphertext),
            "ciphertext_preview": ciphertext[:24].hex(),
            "payload": decoded,
            "wall_clock": time.time(),
        }
        with self._lock:
            self.last_payload = decoded
            self.last_frame = frame
        return frame

    def send_downlink(self, command: dict) -> dict:
        """Seal one command frame from the gateway back to the device."""
        plaintext = json.dumps(command, separators=(",", ":")).encode()
        with self.session._lock:
            seq, ciphertext, opened = self.session.roundtrip_downlink(plaintext)
            epoch = self.session.epoch
            profile = self.session.profile.value if self.session.profile else None
        self.metrics.record_sent(len(plaintext), len(ciphertext))
        self.metrics.record_received()
        return {
            "device_id": self.device_id,
            "seq": seq,
            "epoch": epoch,
            "profile": profile,
            "ciphertext_bytes": len(ciphertext),
            "ciphertext_preview": ciphertext[:24].hex(),
            "command": json.loads(opened),
            "wall_clock": time.time(),
        }

    # --- adversary ----------------------------------------------------------
    def set_attack(self, attack_type: AttackType | None, intensity: float = 0.5, label: str = ""):
        return self.channel.set_attack(attack_type, intensity, label)

    # --- introspection ------------------------------------------------------
    def recent(self, n: int = 120) -> list[dict]:
        with self._lock:
            items = list(self.history)[-n:]
        return [r.as_dict() for r in items]

    def state(self) -> dict:
        with self._lock:
            last = self.history[-1] if self.history else None
        return {
            "device_id": self.device_id,
            "display_name": self.display_name,
            "role": self.config.role,
            "uptime_s": round(time.time() - self.created_at, 1),
            "rounds": self.channel.t,
            "rounds_per_second": self.config.rounds_per_second,
            "n_qubits_per_round": self.config.n_qubits_per_round,
            "qber": last.qber if last else None,
            "confidence": last.confidence if last else 0.0,
            "profile": self.session.profile.value if self.session.profile else None,
            "escalate_threshold": self.escalate_threshold,
            "de_escalate_threshold": self.de_escalate_threshold,
            "cooldown_rounds": self.config.cooldown_rounds,
            "detector_backend": self.detector.backend,
            "detector_warm": len(self.qber_tail) >= self.detector.window_size,
            "predicted_type": self.predicted_type,
            "predicted_type_confidence": round(self.predicted_type_confidence, 3),
            "eve": self.channel.eve.as_dict(),
            "session": self.session.state(),
            # Monotonic totals -- authoritative for any counter shown in a UI.
            "n_escalations": self.n_escalations,
            "n_de_escalations": self.n_de_escalations,
            # Most recent rekeys, so a console opened mid-session reconstructs
            # recent history. Capped for payload size: count from the fields
            # above, never from the length of this list.
            "handshakes": [h.as_dict() for h in self.session.handshakes[-50:]],
            "metrics": self.metrics.snapshot(),
            "last_frame": self.last_frame,
            "connected_sensor": self.connected_sensor,
            "connected_monitor": self.connected_monitor,
        }
