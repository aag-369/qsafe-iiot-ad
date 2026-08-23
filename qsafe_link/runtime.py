"""
The demo runtime: owns every edge node, drives their control loops, and
publishes a single ordered event stream that the UIs render.

Threading model
---------------
All node stepping happens on **one** worker thread, round-robin. This is
deliberate rather than incidental:

  * Qiskit Aer's simulator is a module-level singleton in `qkd_sim.bb84`,
    and the TFLite interpreter in `LiveDetector` is not re-entrant. One
    stepping thread makes both safe by construction instead of by hoping.
  * It keeps ordering deterministic, so the console's event stream is a
    faithful record of what happened rather than a race.

The worker publishes into a bounded deque; an asyncio task in `gateway.py`
drains it and fans out to WebSocket clients. Nothing in the control loop
ever blocks on a network client, so a phone with a bad connection cannot
stall the physics.

Fleet correlation
-----------------
`fleet.correlator.FleetCorrelator` expects equal-length, round-indexed
arrays. Live devices join at different times, so alignment is by *recency*:
the last K rounds of each device, K being the shortest history in the fleet.
Because every node is stepped by the same loop at the same nominal rate,
index alignment tracks time alignment closely; the assumption is recorded
here so it is not mistaken for exact timestamp correlation.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from crypto_agility.kem_backend import KEMProfile, get_kem_backend, is_liboqs_available
from fleet.correlator import FleetCorrelator, FleetCorrelatorConfig
from qkd_sim.qber_stream_multiclass import ATTACK_TYPE_NAMES, AttackType

from .detector import LiveDetector, LiveTypeTagger
from .node import EdgeNode, NodeConfig

_CAMPAIGN_MERGE_SECONDS = 12.0
_TYPE_TAG_EVERY_N_TICKS = 8
_FLEET_CORRELATE_EVERY_N_TICKS = 8
_FLEET_WINDOW_ROUNDS = 90


class LinkRuntime:
    def __init__(
        self,
        models_dir: str | Path = "models",
        n_qubits_per_round: int = 64,
        rounds_per_second: float = 4.0,
        min_devices_for_alert: int = 3,
        enable_type_tagger: bool = True,
        idle_pause_after_s: float = 0.0,
    ):
        self.models_dir = Path(models_dir)
        self.n_qubits_per_round = n_qubits_per_round
        self.rounds_per_second = rounds_per_second
        self.min_devices_for_alert = min_devices_for_alert
        self._enable_type_tagger = enable_type_tagger

        # Everything heavy -- TensorFlow, the TFLite interpreter, the liboqs
        # handles, and the startup KEM measurements -- is deferred to the
        # first request that actually needs it. When this app is mounted
        # beside the research dashboard on a small hosted instance, a visitor
        # who never opens the demo should not pay for loading it, and the
        # platform's health check must not trigger a multi-second warm-up.
        self._loaded = False
        self._load_lock = threading.Lock()
        self._detector: LiveDetector | None = None
        self._type_tagger: LiveTypeTagger | None = None
        self.type_tagger_error: str | None = None
        self._kem_backend = None
        self.liboqs = False
        self.hqc_reference_ms = 0.0
        self.bike_reference_ms = 0.0
        self._correlator: FleetCorrelator | None = None

        self.nodes: dict[str, EdgeNode] = {}
        self.events: deque = deque(maxlen=4000)
        # `fleet_alerts` is a bounded display buffer; `n_fleet_alerts` is the
        # authoritative count, since the buffer is trimmed for payload size.
        self.fleet_alerts: list[dict] = []
        self.n_fleet_alerts = 0
        self.active_scenario: str | None = None

        self._nodes_lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_alive = False
        self._last_tick_at = time.time()
        self._last_client_at = time.time()
        # Seconds with no connected client before the control loop stops
        # stepping. 0 disables the pause entirely (the default for a local
        # demo, where the operator may stare at the console before touching
        # anything). See `note_client_activity`.
        self.idle_pause_after_s = idle_pause_after_s
        self._tick = 0
        self.started_at = time.time()
        self.recorder = None  # set by SessionRecorder.attach()

    # --- lazy loading -------------------------------------------------------
    def ensure_loaded(self) -> None:
        """Load models and resolve the KEM backend, exactly once.

        Idempotent and thread-safe: the web layer and the control-loop worker
        can both reach it.
        """
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self._detector = LiveDetector(self.models_dir)
            self._kem_backend = get_kem_backend()
            self.liboqs = (
                is_liboqs_available()
                and type(self._kem_backend).__name__ == "LiboqsKEMBackend"
            )
            if self._enable_type_tagger:
                try:
                    self._type_tagger = LiveTypeTagger(self.models_dir)
                except Exception as exc:
                    self.type_tagger_error = str(exc)
            else:
                self.type_tagger_error = "disabled"

            # Measure, rather than assume, what one handshake costs at each
            # profile on this machine. Both "CPU saved" figures are priced
            # against these, so the counterfactuals are calibrated to the
            # hardware the demo is actually running on rather than to a table.
            self.hqc_reference_ms = self._measure_reference(KEMProfile.HQC_128)
            self.bike_reference_ms = self._measure_reference(KEMProfile.BIKE_L1)

            self._correlator = FleetCorrelator(
                FleetCorrelatorConfig(
                    min_devices=self.min_devices_for_alert,
                    confidence_threshold=self._detector.threshold,
                    merge_gap_rounds=3,
                )
            )
            self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def detector(self) -> LiveDetector:
        self.ensure_loaded()
        assert self._detector is not None
        return self._detector

    @property
    def type_tagger(self) -> LiveTypeTagger | None:
        self.ensure_loaded()
        return self._type_tagger

    @property
    def kem_backend(self):
        self.ensure_loaded()
        return self._kem_backend

    @property
    def correlator(self) -> FleetCorrelator:
        self.ensure_loaded()
        assert self._correlator is not None
        return self._correlator

    # --- setup --------------------------------------------------------------
    def _measure_reference(self, profile: KEMProfile, n: int = 5) -> float:
        """Median cost of one full handshake at `profile` on this host.

        Called from inside `ensure_loaded()`, so it reaches for the backend
        handle directly rather than the property, which would recurse.
        """
        samples = []
        backend = self._kem_backend
        for _ in range(n):
            pk, sk, kg = backend.keygen(profile)
            ct, _, enc = backend.encapsulate(profile, pk)
            _, dec = backend.decapsulate(profile, sk, ct)
            samples.append(kg.latency_ms + enc.latency_ms + dec.latency_ms)
        return statistics.median(samples)

    def add_node(
        self,
        device_id: str,
        display_name: str = "",
        role: str = "field-sensor",
        rounds_per_second: float | None = None,
        n_qubits_per_round: int | None = None,
        seed: int | None = None,
    ) -> EdgeNode:
        self.ensure_loaded()
        with self._nodes_lock:
            if device_id in self.nodes:
                return self.nodes[device_id]
            cfg = NodeConfig(
                device_id=device_id,
                display_name=display_name or device_id,
                role=role,
                n_qubits_per_round=n_qubits_per_round or self.n_qubits_per_round,
                rounds_per_second=rounds_per_second or self.rounds_per_second,
                seed=seed,
            )
            node = EdgeNode(
                cfg,
                self.detector,
                self.kem_backend,
                hqc_reference_ms=self.hqc_reference_ms,
                bike_reference_ms=self.bike_reference_ms,
            )
            self.nodes[device_id] = node
        self.emit(
            "device_added",
            {"device_id": device_id, "display_name": node.display_name, "role": role},
        )
        return node

    def remove_node(self, device_id: str) -> bool:
        with self._nodes_lock:
            node = self.nodes.pop(device_id, None)
        if node is None:
            return False
        self.emit("device_removed", {"device_id": device_id})
        return True

    def get(self, device_id: str) -> EdgeNode | None:
        with self._nodes_lock:
            return self.nodes.get(device_id)

    def node_list(self) -> list[EdgeNode]:
        with self._nodes_lock:
            return list(self.nodes.values())

    # --- idle handling ------------------------------------------------------
    def note_client_activity(self) -> None:
        """Called whenever a browser client connects or is present.

        The control loop executes real Qiskit circuits continuously, which is
        fine on a laptop during a demonstration and wasteful on a small
        always-on hosted instance where the page may be open to nobody for
        days. Rather than stopping the worker (which would lose state and
        make the first click after a pause feel broken), it simply stops
        stepping while no client is watching, and resumes instantly on the
        next connection.
        """
        self._last_client_at = time.time()

    @property
    def idle(self) -> bool:
        if self.idle_pause_after_s <= 0:
            return False
        return (time.time() - self._last_client_at) > self.idle_pause_after_s

    # --- event bus ----------------------------------------------------------
    def emit(self, kind: str, data: dict) -> None:
        event = {"kind": kind, "ts": time.time(), "data": data}
        self.events.append(event)
        if self.recorder is not None:
            try:
                self.recorder.on_event(event)
            except Exception:
                pass

    # --- worker -------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._worker_alive = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="qsafe-link-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        # An unhandled exception here would end the worker thread silently:
        # the UI would keep rendering its last snapshot with a ticking clock
        # and no error anywhere. Everything in the loop body is guarded.
        while not self._stop.is_set():
            try:
                self._tick_once()
            except Exception as exc:
                self.emit("worker_error", {"error": str(exc)})
            time.sleep(0.005)
        self._worker_alive = False

    def _tick_once(self) -> None:
        """One pass over every device that is due. Never raises."""
        if self.idle:
            # Nobody is watching. Hold every device where it is rather than
            # executing quantum circuits into the void.
            return
        now = time.time()
        stepped = False
        for node in self.node_list():
            if not node.due(now):
                continue
            stepped = True
            try:
                rec = node.step()
            except Exception as exc:  # never let one device kill the loop
                self.emit("device_error", {"device_id": node.device_id, "error": str(exc)})
                continue

            payload = rec.as_dict()
            payload["device_id"] = node.device_id
            self.emit("round", payload)

            if rec.escalated or rec.de_escalated:
                try:
                    self.emit(
                        "rekey",
                        {
                            "device_id": node.device_id,
                            "display_name": node.display_name,
                            "direction": "escalate" if rec.escalated else "de-escalate",
                            **node.session.last_handshake.as_dict(),
                        },
                    )
                except Exception as exc:
                    self.emit("device_error",
                              {"device_id": node.device_id, "error": str(exc)})

        if stepped:
            self._tick += 1
            if self._tick % _TYPE_TAG_EVERY_N_TICKS == 0:
                try:
                    self._run_type_tagging()
                except Exception as exc:
                    self.emit("type_tagger_error", {"error": str(exc)})
            if self._tick % _FLEET_CORRELATE_EVERY_N_TICKS == 0:
                try:
                    self._run_fleet_correlation()
                except Exception as exc:
                    self.emit("correlator_error", {"error": str(exc)})
        self._last_tick_at = time.time()

    # --- additive layers ----------------------------------------------------
    def _run_type_tagging(self) -> None:
        """Batch the attack-type classifier across all warm devices.

        Strictly additive: the result is a caption. It is never fed back into
        `SwitchController`, so a misclassified type cannot weaken the active
        cryptographic posture -- the property the paper's Fig. 1 calls out.
        """
        if self.type_tagger is None:
            return
        nodes, windows = [], []
        for node in self.node_list():
            w = self.type_tagger.build_window(list(node.qber_tail))
            if w is not None:
                nodes.append(node)
                windows.append(w)
        if not windows:
            return
        try:
            results = self.type_tagger.tag_batch(windows)
        except Exception as exc:
            self.emit("type_tagger_error", {"error": str(exc)})
            return
        for node, (idx, conf) in zip(nodes, results):
            name = ATTACK_TYPE_NAMES[AttackType(idx)]
            if name != node.predicted_type:
                self.emit(
                    "type_change",
                    {
                        "device_id": node.device_id,
                        "from": node.predicted_type,
                        "to": name,
                        "confidence": round(conf, 3),
                    },
                )
            node.predicted_type = name
            node.predicted_type_confidence = conf

    def _run_fleet_correlation(self) -> None:
        nodes = self.node_list()
        # The device-count guard is independent of the alert threshold: with
        # `--min-devices-alert 0` the threshold check passes with an empty
        # fleet and min() over no histories raises.
        if len(nodes) < 2 or len(nodes) < self.min_devices_for_alert:
            return
        histories = {n.device_id: n.recent(_FLEET_WINDOW_ROUNDS) for n in nodes}
        if not histories:
            return
        k = min(len(h) for h in histories.values())
        if k < 10:
            return

        type_index = {v: int(k_) for k_, v in ATTACK_TYPE_NAMES.items()}
        confidences, types = {}, {}
        for dev_id, hist in histories.items():
            tail = hist[-k:]
            confidences[dev_id] = np.array([r["confidence"] for r in tail], dtype=float)
            types[dev_id] = np.array(
                [type_index.get(r["predicted_type"], 0) for r in tail], dtype=int
            )

        result = self.correlator.analyze(confidences, types)
        alerts = [
            {
                "t_start": a.t_start,
                "t_end": a.t_end,
                "device_ids": a.device_ids,
                "peak_device_count": int(a.peak_device_count),
                "dominant_attack_type": a.dominant_attack_type,
                "type_agreement": round(a.type_agreement, 3),
                "window_rounds": k,
                "ts": time.time(),
            }
            for a in result.alerts
        ]
        # Announce a genuinely new campaign, not the same one every tick.
        # Keying on the exact device set was too strict: during one sustained
        # campaign the elevated set churns round to round as individual
        # confidences dip, which minted a "new" alert roughly every tick.
        # Treat an alert as a continuation if it shares the attack type and
        # any device with the previous one, within a short window.
        if alerts:
            newest = alerts[-1]
            prev = self.fleet_alerts[-1] if self.fleet_alerts else None
            is_continuation = False
            if prev is not None:
                same_type = prev["dominant_attack_type"] == newest["dominant_attack_type"]
                overlap = set(prev["device_ids"]) & set(newest["device_ids"])
                recent = (newest["ts"] - prev["ts"]) < _CAMPAIGN_MERGE_SECONDS
                is_continuation = same_type and bool(overlap) and recent
            if is_continuation:
                # Fold into the standing alert so the count stays truthful.
                prev["t_end"] = newest["t_end"]
                prev["ts"] = newest["ts"]
                prev["peak_device_count"] = max(
                    prev["peak_device_count"], newest["peak_device_count"]
                )
                prev["device_ids"] = sorted(
                    set(prev["device_ids"]) | set(newest["device_ids"])
                )
            else:
                self.fleet_alerts.append(newest)
                self.n_fleet_alerts += 1
                self.emit("fleet_alert", newest)

    # --- introspection ------------------------------------------------------
    def fleet_state(self) -> dict:
        nodes = self.node_list()
        escalated = [n for n in nodes if n.session.profile == KEMProfile.HQC_128]
        total_adaptive = sum(n.metrics.adaptive_kem_ms for n in nodes)
        total_static = sum(n.metrics.static_hqc_kem_ms for n in nodes)
        round_adaptive = sum(n.metrics.round_adaptive_ms for n in nodes)
        round_static = sum(n.metrics.round_static_ms for n in nodes)
        total_rounds = sum(sum(n.metrics.rounds_by_profile.values()) for n in nodes)
        baseline_rounds = sum(n.metrics.rounds_by_profile.get("BIKE-L1", 0) for n in nodes)
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "n_devices": len(nodes),
            "n_escalated": len(escalated),
            "escalated_device_ids": [n.device_id for n in escalated],
            "min_devices_for_alert": self.min_devices_for_alert,
            "fleet_alerts": self.fleet_alerts[-5:],
            "n_fleet_alerts": self.n_fleet_alerts,
            "active_scenario": self.active_scenario,
            "totals": {
                "adaptive_kem_ms": round(total_adaptive, 2),
                "static_hqc_kem_ms": round(total_static, 2),
                "cpu_saved_ms": round(max(0.0, total_static - total_adaptive), 2),
                "cpu_saved_pct": round(
                    (1 - total_adaptive / total_static) * 100 if total_static else 0.0, 2
                ),
                "frames_sent": sum(n.metrics.frames_sent for n in nodes),
                "n_handshakes": sum(n.metrics.n_handshakes for n in nodes),
                "n_escalations": sum(n.n_escalations for n in nodes),
                "n_de_escalations": sum(n.n_de_escalations for n in nodes),
                "round_adaptive_ms": round(round_adaptive, 2),
                "round_static_ms": round(round_static, 2),
                "paper_equivalent_saved_pct": round(
                    (1 - round_adaptive / round_static) * 100 if round_static else 0.0, 2
                ),
                "total_rounds": total_rounds,
                "baseline_round_fraction": round(
                    baseline_rounds / total_rounds if total_rounds else 0.0, 4
                ),
            },
        }

    def backend_state(self, force_load: bool = True) -> dict:
        """Describe the runtime.

        `force_load=False` answers without triggering the model load, so a
        platform health check can poll this endpoint every few seconds
        without waking TensorFlow on an instance nobody is using.
        """
        common = {
            "worker_alive": self._worker_alive
            and bool(self._thread and self._thread.is_alive()),
            "seconds_since_tick": round(time.time() - self._last_tick_at, 2),
            "models_loaded": self._loaded,
            "idle": self.idle,
            "idle_pause_after_s": self.idle_pause_after_s,
            "n_qubits_per_round": self.n_qubits_per_round,
            "rounds_per_second": self.rounds_per_second,
        }
        if not force_load and not self._loaded:
            return {
                **common,
                "kem_backend": "not loaded",
                "liboqs_available": is_liboqs_available(),
                "using_real_liboqs": None,
                "detector_backend": "not loaded",
                "detector_threshold": None,
                "detector_window": None,
                "type_tagger": "not loaded",
                "hqc_reference_ms": 0.0,
                "bike_reference_ms": 0.0,
            }
        return {
            **common,
            "kem_backend": type(self.kem_backend).__name__,
            "liboqs_available": is_liboqs_available(),
            "using_real_liboqs": self.liboqs,
            "detector_backend": self.detector.backend,
            "detector_threshold": self.detector.threshold,
            "detector_window": self.detector.window_size,
            "type_tagger": "loaded" if self.type_tagger else f"unavailable: {self.type_tagger_error}",
            "hqc_reference_ms": round(self.hqc_reference_ms, 3),
            "bike_reference_ms": round(self.bike_reference_ms, 3),
        }
