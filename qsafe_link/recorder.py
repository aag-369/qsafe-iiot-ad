"""
Session recording and the evidence pack.

A live demo is persuasive in the room and worthless afterwards unless it
leaves something behind. `SessionRecorder` captures every event the runtime
emits and turns a session into:

  * `demo_report.json` -- the measured record: detection latencies, rekey
    log with real handshake costs, per-profile round counts, and the
    per-round CPU figure computed the same way the paper computes it.
  * `demo_session.json` -- the raw event trace, replayable offline.

Detection latency is measured honestly. It is the wall-clock gap between the
adversary actually being switched on for a device and that device's switch
controller reaching HQC-128 -- not the gap to the detector's confidence
crossing threshold, which would quietly omit the controller's hysteresis.
The paper reports a median of one round to escalate; this measures the same
quantity live, including the rekey.

Every escalation falls into exactly one of four buckets, and conflating any
two of them would misreport the result:

  * **Detection** -- the adversary was switched on while the device was on
    the baseline profile, and this is the escalation that answered it. Only
    these carry a latency.
  * **Re-acquisition** -- the adversary is still active, but the link had
    briefly dropped back (the stealthy scenarios sit near the threshold and
    do flap). This is the *same* episode being re-detected, so it is neither
    a new detection nor a false alarm.
  * **Unmeasurable episode** -- an attack applied to a device that was
    already hardened. There is no escalation to time. Arming a timer here
    and letting a later flap stop it would turn a mid-attack oscillation
    into a fabricated multi-second "detection", which is exactly what a
    judge mashing scenario buttons produces.
  * **False positive** -- an escalation with no adversary active at all.

Only the last is a false alarm, and it is reported as one.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path


class SessionRecorder:
    def __init__(self, max_events: int = 200_000):
        self.events: list[dict] = []
        self.max_events = max_events
        self.started_at = time.time()
        # device_id -> wall clock at which a *measurable* attack episode began
        self._attack_started: dict[str, float] = {}
        self._attack_kind: dict[str, str] = {}
        self.detections: list[dict] = []
        self.rekeys: list[dict] = []
        self.spurious_escalations = 0
        self.unmeasurable_episodes = 0
        self.re_acquisitions = 0
        # Devices with an adversary currently switched on. Distinct from
        # `_attack_started`, which is consumed by the first escalation.
        self._adversary_active: set[str] = set()
        self._runtime = None

    def attach(self, runtime) -> "SessionRecorder":
        runtime.recorder = self
        self._runtime = runtime
        return self

    def _already_hardened(self, device_id: str) -> bool:
        """Is this device already on the hardened profile right now?"""
        if self._runtime is None:
            return False
        node = self._runtime.get(device_id)
        if node is None or node.session.profile is None:
            return False
        return node.session.profile.value == "HQC-128"

    def _arm(self, device_id: str, kind: str, ts: float) -> None:
        self._adversary_active.add(device_id)
        if self._already_hardened(device_id):
            # Nothing to time. Clear any stale arm so a later flap cannot be
            # mistaken for the detection of this episode.
            self._attack_started.pop(device_id, None)
            self._attack_kind.pop(device_id, None)
            self.unmeasurable_episodes += 1
            return
        self._attack_started[device_id] = ts
        self._attack_kind[device_id] = kind

    def _disarm(self, device_id: str) -> None:
        self._attack_started.pop(device_id, None)
        self._attack_kind.pop(device_id, None)
        self._adversary_active.discard(device_id)

    # --- capture ------------------------------------------------------------
    def on_event(self, event: dict) -> None:
        if len(self.events) < self.max_events:
            self.events.append(event)

        kind = event.get("kind")
        data = event.get("data", {})
        ts = event.get("ts", time.time())

        if kind == "scenario":
            scenario = data.get("scenario", {})
            atype = scenario.get("attack_type", "benign")
            for dev in data.get("devices", []):
                if atype != "benign":
                    self._arm(dev, scenario.get("key", atype), ts)
                else:
                    self._disarm(dev)

        elif kind == "attack_set":
            atype = data.get("attack_type", "benign")
            for dev in data.get("devices", []):
                if atype != "benign":
                    self._arm(dev, atype, ts)
                else:
                    self._disarm(dev)

        elif kind == "rekey":
            dev = data.get("device_id")
            self.rekeys.append({**data, "ts": ts})
            if data.get("direction") != "escalate":
                return
            if dev not in self._attack_started:
                if dev in self._adversary_active:
                    # The adversary is still on; the link had flapped back to
                    # the baseline and has now re-acquired. Same episode, so
                    # neither a new detection nor a false alarm.
                    self.re_acquisitions += 1
                else:
                    # Nothing was attacking: a genuine false positive.
                    self.spurious_escalations += 1
                return
            started = self._attack_started.pop(dev)
            kind_name = self._attack_kind.pop(dev, "unknown")
            self.detections.append(
                {
                    "device_id": dev,
                    "scenario": kind_name,
                    "attack_started_at": started,
                    "escalated_at": ts,
                    "latency_s": round(ts - started, 3),
                    "handshake_ms": data.get("total_ms"),
                    "profile": data.get("profile"),
                }
            )

    # --- report -------------------------------------------------------------
    def build_report(self, runtime) -> dict:
        nodes = runtime.node_list()
        latencies = [d["latency_s"] for d in self.detections]
        escalations = [r for r in self.rekeys if r.get("direction") == "escalate"]
        de_escalations = [r for r in self.rekeys if r.get("direction") == "de-escalate"]

        handshake_by_profile: dict[str, list[float]] = {}
        for r in self.rekeys:
            handshake_by_profile.setdefault(r.get("profile", "?"), []).append(
                float(r.get("total_ms", 0.0))
            )

        device_reports = []
        for node in nodes:
            m = node.metrics.snapshot()
            device_reports.append(
                {
                    "device_id": node.device_id,
                    "display_name": node.display_name,
                    "rounds": node.channel.t,
                    "rounds_by_profile": m["rounds_by_profile"],
                    "baseline_round_fraction": m["baseline_round_fraction"],
                    "paper_equivalent_saved_pct": m["paper_equivalent_saved_pct"],
                    "n_handshakes": m["n_handshakes"],
                    "session_rekey_ms": m["adaptive_kem_ms"],
                    "frames_sent": m["frames_sent"],
                    "frames_rejected": m["frames_rejected"],
                    "plaintext_bytes": m["plaintext_bytes"],
                    "ciphertext_bytes": m["ciphertext_bytes"],
                    "aead_overhead_bytes": m["aead_overhead_bytes"],
                    "final_profile": node.session.profile.value if node.session.profile else None,
                    "epoch": node.session.epoch,
                }
            )

        fleet = runtime.fleet_state()
        backend = runtime.backend_state()

        return {
            "generated_at": time.time(),
            "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "session_duration_s": round(time.time() - self.started_at, 1),
            "backend": backend,
            "provenance": {
                "kem": (
                    "real liboqs (BIKE-L1 / HQC-1)"
                    if backend["using_real_liboqs"]
                    else "SIMULATED KEM backend -- timing-calibrated stand-in, not real cryptography"
                ),
                "detector": (
                    f"{backend['detector_backend']} from committed artifacts, "
                    f"threshold {backend['detector_threshold']:.4f}, window {backend['detector_window']}"
                ),
                "qkd": (
                    "real Qiskit BB84 circuits (state prep, intercept-resend Eve "
                    "conditioned on a mid-circuit measurement, depolarizing noise, sifting)"
                ),
                "note": (
                    "Attack injection sets eve_intercept_prob / channel_error_prob on the "
                    "real BB84 simulation; QBER is measured from the simulated quantum "
                    "channel, never synthesised directly."
                ),
            },
            "detection": {
                "n_episodes": len(self.detections),
                "n_unmeasurable_episodes": self.unmeasurable_episodes,
                "n_re_acquisitions": self.re_acquisitions,
                "n_spurious_escalations": self.spurious_escalations,
                "measurement_note": (
                    "Latency is wall-clock from adversary activation to the hardened "
                    "profile being live, including the rekey. Every escalation falls "
                    "into exactly one bucket: a detection (carries a latency); a "
                    "re-acquisition (adversary still active, link had briefly flapped "
                    "back -- same episode, not a new alarm); an unmeasurable episode "
                    "(device already hardened when the attack was applied, so there "
                    "is no escalation to time); or a false positive (no adversary "
                    "active at all). Only the last is a false alarm."
                ),
                "median_latency_s": round(statistics.median(latencies), 3) if latencies else None,
                "mean_latency_s": round(statistics.mean(latencies), 3) if latencies else None,
                "min_latency_s": round(min(latencies), 3) if latencies else None,
                "max_latency_s": round(max(latencies), 3) if latencies else None,
                "episodes": self.detections,
            },
            "crypto_agility": {
                "n_escalations": len(escalations),
                "n_de_escalations": len(de_escalations),
                "handshake_ms_by_profile": {
                    p: {
                        "n": len(v),
                        "median_ms": round(statistics.median(v), 3) if v else 0.0,
                        "total_ms": round(sum(v), 3),
                    }
                    for p, v in handshake_by_profile.items()
                },
                "rekey_log": self.rekeys,
            },
            "devices": device_reports,
            "fleet": {
                "n_devices": fleet["n_devices"],
                "min_devices_for_alert": fleet["min_devices_for_alert"],
                "n_campaign_alerts": getattr(runtime, "n_fleet_alerts", len(runtime.fleet_alerts)),
                "alerts": runtime.fleet_alerts,
                "totals": fleet["totals"],
            },
            "comparison_to_paper": {
                "paper_detector_f1": 0.936,
                "paper_roc_auc": 0.992,
                "paper_cpu_reduction_pct": 73.2,
                "live_paper_equivalent_saved_pct": fleet["totals"].get(
                    "paper_equivalent_saved_pct"
                ),
                "live_baseline_round_fraction": fleet["totals"].get(
                    "baseline_round_fraction"
                ),
                "explanation": (
                    "live_paper_equivalent_saved_pct applies the published methodology "
                    "(one KEM handshake per round at the active profile, versus one "
                    "HQC-128 handshake per round) to this live session, using "
                    "per-profile handshake costs measured on this host. It converges "
                    "toward the published 73.2% as the live attack duty cycle "
                    "approaches that of the 2,000-round test stream (15.2% attack rounds)."
                ),
            },
        }

    # --- persistence --------------------------------------------------------
    def save(self, runtime, out_dir: str | Path = "demo_output") -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        report_path = out / "demo_report.json"
        report = self.build_report(runtime)
        report_path.write_text(json.dumps(report, indent=2))

        trace_path = out / "demo_session.json"
        trace_path.write_text(
            json.dumps(
                {
                    "started_at": self.started_at,
                    "backend": runtime.backend_state(),
                    "events": self.events,
                },
                indent=2,
            )
        )
        return {"report": report_path, "trace": trace_path}
