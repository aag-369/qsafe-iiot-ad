"""Tests for the live demo runtime (qsafe_link/).

These exercise the demo's own wiring. The layers it wraps (BB84, the GRU,
the switch controller, the KEM backend) have their own tests; what is
verified here is that the demo drives them faithfully -- in particular that
live, streaming inference produces the *same decisions* as the batch
pipeline the published results came from, and that a shared detector cannot
leak one device's history into another's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_agility.kem_backend import KEMProfile, get_kem_backend
from qkd_sim.qber_stream_multiclass import ATTACK_PROFILES, AttackType
from qsafe_link.channel import ChannelConfig, LiveQKDChannel
from qsafe_link.detector import LiveDetector
from qsafe_link.node import EdgeNode, NodeConfig
from qsafe_link.scenarios import SCENARIOS, resolve_attack_type

REPO_MODELS = "models"


@pytest.fixture(scope="module")
def detector():
    return LiveDetector(models_dir=REPO_MODELS)


@pytest.fixture
def node(detector):
    return EdgeNode(
        NodeConfig(device_id="test-01", display_name="Test Node", seed=42),
        detector,
        get_kem_backend(force_simulated=True),
        hqc_reference_ms=9.0,
        bike_reference_ms=0.8,
    )


# --- detector fidelity ----------------------------------------------------
def _synthetic_qber_series(n: int = 300, seed: int = 7) -> np.ndarray:
    """A deterministic QBER series spanning the regimes a real stream has.

    Generated here rather than read from `data/qber_test.csv` on purpose:
    that file is gitignored (large, and regenerable from a fixed seed), so a
    test that depends on it passes on a machine that has run the pipeline and
    fails on every clean checkout, including CI.

    The property under test -- that streaming feature construction and
    inference reach the same decisions as the batch path -- depends only on
    *a* QBER series, not on it having come from Qiskit. The real stream is
    still exercised by the test below, when it happens to be present.
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(0.015, 0.022, n)                       # benign noise floor
    for start, length, level in ((60, 40, 0.16),          # loud interception
                                 (150, 30, 0.05),         # stealthy, near-threshold
                                 (230, 35, 0.21)):        # jamming-scale
        q[start:start + length] = rng.normal(level, level * 0.35, length)
    return np.clip(q, 0.0, 1.0)


def _assert_streaming_matches_batch(detector, qber: np.ndarray) -> None:
    """Streaming inference must reach the same decisions as the batch path.

    Confidences are compared numerically, and decisions are required to agree
    everywhere except within the numeric tolerance of the threshold. INT8
    quantization may tie-break a window sitting on the threshold either way --
    and that arithmetic can differ across TensorFlow builds -- but it must
    never flip a *confident* decision. Asserting exact equality instead would
    be asserting a property of one machine's TF build.
    """
    from orchestrator.detector_runner import DetectorRunner

    df = pd.DataFrame({"qber": qber, "label": 0})
    batch = DetectorRunner(
        model_path=f"{REPO_MODELS}/gru_detector.keras",
        norm_stats_path=f"{REPO_MODELS}/norm_stats.json",
    ).score_stream(df)

    tail, live = [], []
    for q in qber:
        tail.append(float(q))
        live.append(detector.score_tail(tail))
    live = np.array(live)

    warm = np.arange(len(qber)) >= detector.window_size - 1
    lb, bb = live[warm], batch[warm]
    tol, th = 0.01, detector.threshold

    max_dev = np.abs(lb - bb).max()
    assert max_dev < tol, f"confidences diverge by {max_dev:.4f} (>= {tol})"

    disagree = (lb >= th) != (bb >= th)
    margin = np.minimum(np.abs(lb - th), np.abs(bb - th))
    confident_flips = int((disagree & (margin > tol)).sum())
    assert confident_flips == 0, (
        f"{confident_flips} confident decision(s) flipped between the streaming "
        f"and batch paths -- quantization may only tie-break borderline windows"
    )
    assert disagree.mean() < 0.02, (
        f"{disagree.mean():.1%} of decisions disagree; expected < 2% even at the threshold"
    )


def test_live_detector_matches_batch_pipeline(detector):
    """The whole demo rests on this: streaming inference must reach the same
    decisions as `orchestrator/detector_runner.py`, which produced the
    published numbers."""
    _assert_streaming_matches_batch(detector, _synthetic_qber_series())


@pytest.mark.skipif(
    not Path("data/qber_test.csv").exists(),
    reason="data/qber_test.csv is gitignored; regenerate with qkd_sim.qber_stream",
)
def test_live_detector_matches_batch_pipeline_on_the_real_stream(detector):
    """Same property, against the actual held-out BB84 stream when it exists."""
    qber = pd.read_csv("data/qber_test.csv").head(250)["qber"].to_numpy()
    _assert_streaming_matches_batch(detector, qber)


def test_detector_is_cold_until_window_fills(detector):
    for n in range(1, detector.window_size):
        assert detector.score_tail([0.5] * n) == 0.0
    assert detector.score_tail([0.5] * detector.window_size) >= 0.0


def test_shared_detector_does_not_leak_between_devices(detector):
    """A shared detector must be stateless: two devices scoring alternately
    must get the same answers as if each ran alone."""
    a_stream = [0.02 + 0.001 * i for i in range(60)]
    b_stream = [0.30] * 60

    solo_a = []
    tail = []
    for q in a_stream:
        tail.append(q)
        solo_a.append(detector.score_tail(tail))

    ta, tb, inter_a = [], [], []
    for qa, qb in zip(a_stream, b_stream):
        ta.append(qa)
        inter_a.append(detector.score_tail(ta))
        tb.append(qb)
        detector.score_tail(tb)

    assert np.allclose(solo_a, inter_a, atol=1e-6)


def test_detector_reports_its_backend(detector):
    assert detector.backend in ("tflite-int8", "keras-fp32")
    assert detector.describe()["threshold"] == detector.threshold


# --- channel --------------------------------------------------------------
def test_benign_channel_stays_near_the_noise_floor():
    ch = LiveQKDChannel(ChannelConfig(n_qubits_per_round=64, seed=7))
    qber = [ch.step().qber for _ in range(40)]
    assert np.mean(qber) < 0.05


def test_attack_raises_qber_above_benign():
    ch = LiveQKDChannel(ChannelConfig(n_qubits_per_round=64, seed=7))
    benign = np.mean([ch.step().qber for _ in range(30)])
    ch.set_attack(AttackType.EAVESDROP, 0.9)
    attacked = np.mean([ch.step().qber for _ in range(30)])
    assert attacked > benign * 2


def test_attack_parameters_come_from_the_published_table():
    """Live attacks must be parameterised from the same table the
    attack-type classifier was trained against, not from new constants."""
    ch = LiveQKDChannel(ChannelConfig(seed=1))
    for atype in (AttackType.EAVESDROP, AttackType.JAMMING, AttackType.PNS):
        profile = ATTACK_PROFILES[atype]
        for intensity in (0.0, 0.5, 1.0):
            eve = ch.set_attack(atype, intensity)
            lo, hi = profile["eve_intercept_prob_range"]
            assert lo - 1e-9 <= eve.eve_intercept_prob <= hi + 1e-9
            lo, hi = profile["channel_error_prob_range"]
            assert lo - 1e-9 <= eve.channel_error_prob <= hi + 1e-9


def test_jamming_does_not_intercept():
    ch = LiveQKDChannel(ChannelConfig(seed=1))
    eve = ch.set_attack(AttackType.JAMMING, 1.0)
    assert eve.eve_intercept_prob == 0.0
    assert eve.channel_error_prob > 0.1


def test_clear_attack_restores_benign_state():
    ch = LiveQKDChannel(ChannelConfig(seed=1))
    ch.set_attack(AttackType.EAVESDROP, 0.8)
    assert ch.eve.active
    assert not ch.clear_attack().active


def test_ground_truth_is_recorded_on_every_sample():
    ch = LiveQKDChannel(ChannelConfig(n_qubits_per_round=32, seed=1))
    assert ch.step().ground_truth_attack is False
    ch.set_attack(AttackType.JAMMING, 0.7)
    s = ch.step()
    assert s.ground_truth_attack is True
    assert s.ground_truth_type == "jamming"


# --- node -----------------------------------------------------------------
def test_node_starts_on_the_baseline_profile(node):
    assert node.session.profile is KEMProfile.BIKE_L1
    assert node.session.epoch == 0


def test_node_escalates_under_attack_and_rekeys(node):
    for _ in range(25):
        node.step()
    epoch_before = node.session.epoch
    node.set_attack(AttackType.JAMMING, 0.8)
    for _ in range(30):
        rec = node.step()
        if rec.profile == "HQC-128":
            break
    assert node.session.profile is KEMProfile.HQC_128
    assert node.session.epoch > epoch_before, "escalation must trigger a real rekey"
    assert node.session.last_handshake.profile == "HQC-128"


def test_node_de_escalates_after_the_attack_clears(node):
    for _ in range(25):
        node.step()
    node.set_attack(AttackType.JAMMING, 0.9)
    for _ in range(30):
        node.step()
        if node.session.profile is KEMProfile.HQC_128:
            break
    assert node.session.profile is KEMProfile.HQC_128
    node.channel.clear_attack()
    for _ in range(60):
        node.step()
        if node.session.profile is KEMProfile.BIKE_L1:
            break
    assert node.session.profile is KEMProfile.BIKE_L1


def test_uplink_frames_round_trip_and_expand_by_the_tag(node):
    frame = node.send_uplink({"pressure_bar": 4.2})
    assert frame["payload"] == {"pressure_bar": 4.2}
    assert frame["ciphertext_bytes"] == frame["plaintext_bytes"] + 16
    assert len(frame["ciphertext_preview"]) == 48  # 24 bytes as hex


def test_traffic_survives_a_rekey(node):
    node.send_uplink({"n": 1})
    node.session.rekey(KEMProfile.HQC_128, reason="escalate")
    frame = node.send_uplink({"n": 2})
    assert frame["payload"] == {"n": 2}
    assert frame["epoch"] == node.session.epoch


def test_downlink_commands_round_trip(node):
    frame = node.send_downlink({"command": "setpoint", "value": 40})
    assert frame["command"]["value"] == 40


def test_node_uses_the_published_threshold(node, detector):
    assert node.escalate_threshold == detector.threshold
    assert node.de_escalate_threshold < node.escalate_threshold


def test_paper_equivalent_cost_beats_static_when_mostly_benign(node):
    for _ in range(40):
        node.step()
    m = node.metrics.snapshot()
    assert m["baseline_round_fraction"] > 0.5
    assert m["paper_equivalent_saved_pct"] > 0, "adaptive must be cheaper than always-on"


def test_node_state_is_json_serialisable(node):
    import json

    node.step()
    json.dumps(node.state())


# --- scenarios ------------------------------------------------------------
def test_every_scenario_resolves_to_a_real_attack_type():
    for key, sc in SCENARIOS.items():
        if sc.attack_type is None:
            continue
        assert sc.attack_type in ATTACK_PROFILES
        assert 0.0 <= sc.intensity <= 1.0
        assert sc.scope in ("single", "fleet")
        assert sc.expect and sc.detail


def test_calm_scenario_clears_the_adversary():
    assert SCENARIOS["calm"].attack_type is None


def test_resolve_attack_type_rejects_unknown_names():
    assert resolve_attack_type("jamming") is AttackType.JAMMING
    assert resolve_attack_type("benign") is None
    with pytest.raises(ValueError):
        resolve_attack_type("nonsense")


# --- gateway API ----------------------------------------------------------
@pytest.fixture(scope="module")
def client(detector):
    from fastapi.testclient import TestClient

    from qsafe_link.gateway import create_app
    from qsafe_link.recorder import SessionRecorder
    from qsafe_link.runtime import LinkRuntime

    runtime = LinkRuntime(
        models_dir=REPO_MODELS, n_qubits_per_round=16,
        rounds_per_second=2.0, enable_type_tagger=False,
    )
    recorder = SessionRecorder().attach(runtime)
    runtime.add_node("api-01", "API Node", seed=3)
    app = create_app(runtime, recorder=recorder, port=8123)
    with TestClient(app) as c:
        yield c
    runtime.stop()


def test_health_reports_provenance(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "using_real_liboqs" in body
    assert body["detector_backend"] in ("tflite-int8", "keras-fp32")


def test_state_includes_devices_and_fleet_totals(client):
    body = client.get("/api/state").json()
    assert body["devices"] and body["devices"][0]["device_id"] == "api-01"
    assert "paper_equivalent_saved_pct" in body["fleet"]["totals"]


def test_scenarios_endpoint_lists_every_scenario(client):
    body = client.get("/api/scenarios").json()
    assert {s["key"] for s in body["scenarios"]} == set(SCENARIOS)


def test_applying_a_scenario_sets_the_adversary(client):
    body = client.post("/api/scenario", json={"key": "jamming", "device_id": "api-01"}).json()
    assert body["applied_to"] == ["api-01"]
    state = client.get("/api/state").json()
    assert state["devices"][0]["eve"]["active"] is True
    client.post("/api/scenario", json={"key": "calm"})


def test_unknown_scenario_is_rejected(client):
    assert client.post("/api/scenario", json={"key": "nope"}).status_code == 400


def test_unknown_attack_type_is_rejected(client):
    r = client.post("/api/attack", json={"device_id": "api-01", "attack_type": "nope"})
    assert r.status_code == 400


def test_command_endpoint_seals_a_downlink_frame(client):
    body = client.post(
        "/api/command", json={"device_id": "api-01", "command": "setpoint", "value": 55}
    ).json()
    assert body["command"]["value"] == 55
    assert body["ciphertext_bytes"] > 0


def test_command_to_unknown_device_is_404(client):
    r = client.post("/api/command", json={"device_id": "ghost", "command": "x"})
    assert r.status_code == 404


def test_pages_are_served(client):
    for path in ("/", "/console", "/node", "/monitor", "/replay"):
        assert client.get(path).status_code == 200, path


def test_join_urls_are_absolute(client):
    body = client.get("/api/join-urls").json()
    assert body["console"].endswith("/console")
    assert body["node"].startswith("http://")


def test_report_endpoint_declares_provenance(client):
    body = client.get("/api/report").json()
    assert "kem" in body["provenance"]
    assert "comparison_to_paper" in body


def test_device_lifecycle(client):
    client.post("/api/devices", json={"device_id": "api-02", "display_name": "Second"})
    assert any(d["device_id"] == "api-02" for d in client.get("/api/state").json()["devices"])
    assert client.delete("/api/devices/api-02").status_code == 200
    assert client.delete("/api/devices/api-02").status_code == 404


# --- regressions ----------------------------------------------------------
# Each of these encodes a defect found in review that would have produced a
# wrong number on screen or an unrecoverable state during a live demo.

class _FakeSession:
    def __init__(self, profile):
        self.profile = type("P", (), {"value": profile})()


class _FakeNode:
    def __init__(self, profile):
        self.session = _FakeSession(profile)


class _FakeRuntime:
    def __init__(self, profiles):
        self._nodes = {k: _FakeNode(v) for k, v in profiles.items()}
        self.recorder = None
        self.fleet_alerts = []
        self.n_fleet_alerts = 0

    def get(self, device_id):
        return self._nodes.get(device_id)

    def set_profile(self, device_id, profile):
        self._nodes[device_id].session.profile = type("P", (), {"value": profile})()


def _scenario_event(device, key, attack_type, ts):
    return {
        "kind": "scenario", "ts": ts,
        "data": {"scenario": {"key": key, "attack_type": attack_type}, "devices": [device]},
    }


def _rekey_event(device, direction, ts):
    return {
        "kind": "rekey", "ts": ts,
        "data": {"device_id": device, "direction": direction,
                 "profile": "HQC-128" if direction == "escalate" else "BIKE-L1",
                 "total_ms": 9.5},
    }


def test_flap_during_a_sustained_attack_is_not_a_new_detection():
    """Regression: applying a second attack while the device is already
    hardened used to leave a timer armed, so the next mid-attack flap was
    recorded as a detection with a latency of however long the attack had
    been running -- inflating the headline figure by an order of magnitude."""
    from qsafe_link.recorder import SessionRecorder

    rt = _FakeRuntime({"d1": "BIKE-L1"})
    rec = SessionRecorder().attach(rt)

    rec.on_event(_scenario_event("d1", "eavesdrop", "eavesdrop", 0.0))
    rt.set_profile("d1", "HQC-128")
    rec.on_event(_rekey_event("d1", "escalate", 1.2))

    # Judge taps another attack while the device is still hardened.
    rec.on_event(_scenario_event("d1", "campaign", "eavesdrop", 20.0))
    # Mid-attack flap.
    rt.set_profile("d1", "BIKE-L1")
    rec.on_event(_rekey_event("d1", "de-escalate", 55.0))
    rt.set_profile("d1", "HQC-128")
    rec.on_event(_rekey_event("d1", "escalate", 56.0))

    latencies = [d["latency_s"] for d in rec.detections]
    assert latencies == [1.2], f"fabricated detection: {latencies}"
    assert rec.unmeasurable_episodes == 1


def test_escalation_with_no_adversary_is_a_false_positive_not_a_detection():
    from qsafe_link.recorder import SessionRecorder

    rt = _FakeRuntime({"d1": "BIKE-L1"})
    rec = SessionRecorder().attach(rt)
    rec.on_event(_rekey_event("d1", "escalate", 5.0))
    assert rec.detections == []
    assert rec.spurious_escalations == 1


def test_clearing_the_adversary_disarms_the_timer():
    from qsafe_link.recorder import SessionRecorder

    rt = _FakeRuntime({"d1": "BIKE-L1"})
    rec = SessionRecorder().attach(rt)
    rec.on_event(_scenario_event("d1", "hndl", "eavesdrop", 0.0))
    rec.on_event(_scenario_event("d1", "calm", "benign", 3.0))
    rec.on_event(_rekey_event("d1", "escalate", 9.0))
    assert rec.detections == []
    assert rec.spurious_escalations == 1


def test_next_tick_advances_even_when_a_round_raises(node, monkeypatch):
    """Regression: `_next_tick` was set on the last line of step(), so a
    raising round left the node permanently 'due' and the worker retried it
    at full loop speed, flooding the bounded event deque and pushing real
    rounds out of it."""
    def boom():
        raise RuntimeError("simulated Aer failure")

    monkeypatch.setattr(node.channel, "step", boom)
    before = node._next_tick
    with pytest.raises(RuntimeError):
        node.step()
    assert node._next_tick > before
    assert not node.due(node._next_tick - 0.001)


def test_escalation_counters_are_monotonic(node):
    """Regression: the console counted escalations from the capped handshake
    list, so the tile decreased once a device passed the cap."""
    assert node.n_escalations == 0
    for _ in range(25):
        node.step()
    node.set_attack(AttackType.JAMMING, 0.9)
    for _ in range(30):
        node.step()
        if node.session.profile is KEMProfile.HQC_128:
            break
    counted = node.state()["n_escalations"]
    assert counted >= 1
    node.channel.clear_attack()
    for _ in range(60):
        node.step()
    assert node.state()["n_escalations"] >= counted, "counter must never decrease"


def test_cpu_saved_never_reports_a_negative_saving():
    """Regression: the counterfactual charged a reference cost measured on an
    idle process, so under load a hardened link could report a negative
    saving -- rendered in green on the console."""
    from secure_channel.metrics import LinkMetrics

    m = LinkMetrics(hqc_reference_ms=9.0, bike_reference_ms=0.8)
    for _ in range(5):
        m.record_handshake("HQC-128", 25.0)   # far slower than the reference
    assert m.cpu_saved_pct >= 0.0
    assert m.cpu_saved_ms >= 0.0
    assert m.paper_equivalent_saved_pct >= 0.0


def test_metric_dicts_are_preseeded_so_iteration_cannot_race():
    """Regression: both profile dicts gained their HQC-128 key on the first
    escalation. A snapshot iterating at that instant raised
    'dictionary changed size during iteration' -- at the single most
    important moment of the demo."""
    from secure_channel.metrics import LinkMetrics

    m = LinkMetrics()
    assert set(m.rounds_by_profile) == {"BIKE-L1", "HQC-128"}
    assert set(m.handshake_ms_by_profile) == {"BIKE-L1", "HQC-128"}
    before = set(m.rounds_by_profile)
    m.record_round("HQC-128")
    m.record_handshake("HQC-128", 9.0)
    assert set(m.rounds_by_profile) == before


def test_rekey_cannot_split_a_frame_roundtrip(node):
    """Regression: seal and open were separate lock acquisitions, so a rekey
    landing between them failed authentication on a frame that was valid
    when it was created."""
    import threading

    errors = []

    def rekey_storm():
        for _ in range(40):
            try:
                node.session.rekey(
                    KEMProfile.HQC_128 if _ % 2 else KEMProfile.BIKE_L1, reason="churn"
                )
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

    t = threading.Thread(target=rekey_storm)
    t.start()
    for i in range(120):
        try:
            frame = node.send_uplink({"i": i})
            assert frame["payload"] == {"i": i}
        except Exception as exc:
            errors.append(exc)
    t.join()
    assert not errors, f"frames failed across concurrent rekeys: {errors[:3]}"


def test_fleet_correlation_survives_a_zero_alert_threshold(detector):
    """Regression: `--min-devices-alert 0` passed the threshold check with an
    empty fleet, and min() over no histories killed the worker thread."""
    from qsafe_link.runtime import LinkRuntime

    rt = LinkRuntime(models_dir=REPO_MODELS, n_qubits_per_round=16,
                     min_devices_for_alert=0, enable_type_tagger=False)
    rt._run_fleet_correlation()          # no devices at all
    rt.add_node("solo", "Solo", seed=1)
    rt._run_fleet_correlation()          # one device
    rt.stop()


def test_worker_reports_liveness(detector):
    from qsafe_link.runtime import LinkRuntime

    rt = LinkRuntime(models_dir=REPO_MODELS, n_qubits_per_round=16,
                     enable_type_tagger=False)
    assert rt.backend_state()["worker_alive"] is False
    rt.start()
    assert rt.backend_state()["worker_alive"] is True
    rt.stop()


def test_re_escalation_during_a_live_attack_is_not_a_false_positive():
    """Regression: a mid-attack flap re-escalating was counted as a false
    positive, overstating the false-alarm rate in the evidence pack. The
    adversary was still active -- that is a re-acquisition of the same
    episode."""
    from qsafe_link.recorder import SessionRecorder

    rt = _FakeRuntime({"d1": "BIKE-L1"})
    rec = SessionRecorder().attach(rt)

    rec.on_event(_scenario_event("d1", "pns", "pns", 0.0))
    rt.set_profile("d1", "HQC-128")
    rec.on_event(_rekey_event("d1", "escalate", 1.0))       # detection
    rt.set_profile("d1", "BIKE-L1")
    rec.on_event(_rekey_event("d1", "de-escalate", 4.0))    # flap
    rt.set_profile("d1", "HQC-128")
    rec.on_event(_rekey_event("d1", "escalate", 5.0))       # re-acquisition

    assert [d["latency_s"] for d in rec.detections] == [1.0]
    assert rec.re_acquisitions == 1
    assert rec.spurious_escalations == 0, "a live attack must not read as a false alarm"

    # Now clear the adversary; a later escalation IS a false positive.
    rec.on_event(_scenario_event("d1", "calm", "benign", 10.0))
    rec.on_event(_rekey_event("d1", "escalate", 30.0))
    assert rec.spurious_escalations == 1


# --- deployment: mounting under a URL prefix ------------------------------
# The demo is served standalone during a live demonstration and mounted at
# /link on the deployed site. Both must work, and the difference must be
# invisible to the client code.

@pytest.fixture(scope="module")
def mounted_client(detector):
    """The demo mounted under a prefix, as `server.py` deploys it."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from qsafe_link.gateway import create_app
    from qsafe_link.runtime import LinkRuntime

    runtime = LinkRuntime(
        models_dir=REPO_MODELS, n_qubits_per_round=16,
        rounds_per_second=2.0, enable_type_tagger=False,
    )
    link = create_app(runtime, port=8000)

    root = FastAPI()
    root.mount("/link", link)
    with TestClient(root) as c:
        yield c
    runtime.stop()


def test_pages_resolve_assets_under_the_mount_prefix(mounted_client):
    body = mounted_client.get("/link/console").text
    assert "__BASE__" not in body, "template placeholder survived into the response"
    assert "/link/static/shared.css" in body
    assert "/link/static/qsafe.js" in body
    assert 'window.QSAFE_BASE = "/link"' in body


def test_every_client_page_is_templated(mounted_client):
    for path in ("/link/", "/link/console", "/link/node", "/link/monitor", "/link/replay"):
        body = mounted_client.get(path).text
        assert "__BASE__" not in body, path
        assert 'window.QSAFE_BASE = "/link"' in body, path


def test_static_assets_are_reachable_under_the_prefix(mounted_client):
    assert mounted_client.get("/link/static/qsafe.js").status_code == 200
    assert mounted_client.get("/link/static/shared.css").status_code == 200


def test_join_urls_carry_the_prefix(mounted_client):
    urls = mounted_client.get("/link/api/join-urls").json()
    assert urls["console"].endswith("/link/console")
    assert urls["node"].endswith("/link/node")
    assert "/link" in urls["base"]


def test_served_standalone_the_prefix_is_empty(client):
    body = client.get("/console").text
    assert "__BASE__" not in body
    assert 'window.QSAFE_BASE = ""' in body
    assert '"/static/qsafe.js"' in body or "'/static/qsafe.js'" in body


# --- deployment: lazy loading and idle pause ------------------------------
def test_health_does_not_force_the_model_load():
    """A hosting platform polls health every few seconds. Waking TensorFlow
    for that on an instance nobody is using is pure waste."""
    from fastapi.testclient import TestClient

    from qsafe_link.gateway import create_app
    from qsafe_link.runtime import LinkRuntime

    runtime = LinkRuntime(models_dir=REPO_MODELS, n_qubits_per_round=16,
                          enable_type_tagger=False)
    assert runtime.loaded is False
    with TestClient(create_app(runtime, port=8000)) as c:
        for _ in range(3):
            body = c.get("/api/health").json()
            assert body["models_loaded"] is False
            assert body["status"] == "ok"
        assert runtime.loaded is False, "health checks must not warm the runtime"
    runtime.stop()


def test_adding_a_device_loads_the_models():
    from qsafe_link.runtime import LinkRuntime

    runtime = LinkRuntime(models_dir=REPO_MODELS, n_qubits_per_round=16,
                          enable_type_tagger=False)
    assert runtime.loaded is False
    runtime.add_node("lazy-01", "Lazy")
    assert runtime.loaded is True
    assert runtime.detector.threshold > 0
    runtime.stop()


def test_idle_runtime_stops_stepping():
    """On an always-on instance with nobody watching, the control loop should
    hold rather than execute quantum circuits into the void."""
    import time as _time

    from qsafe_link.runtime import LinkRuntime

    runtime = LinkRuntime(models_dir=REPO_MODELS, n_qubits_per_round=16,
                          rounds_per_second=20.0, enable_type_tagger=False,
                          idle_pause_after_s=0.3)
    node = runtime.add_node("idle-01", "Idle")
    runtime.start()
    _time.sleep(0.6)
    rounds_while_active = node.channel.t
    assert rounds_while_active > 0, "should step while a client is considered present"

    _time.sleep(0.8)              # now past the idle threshold
    assert runtime.idle is True
    paused_at = node.channel.t
    _time.sleep(0.6)
    assert node.channel.t == paused_at, "must not step while idle"

    runtime.note_client_activity()
    assert runtime.idle is False
    _time.sleep(0.5)
    assert node.channel.t > paused_at, "must resume as soon as a client returns"
    runtime.stop()


def test_idle_pause_disabled_by_default(detector):
    from qsafe_link.runtime import LinkRuntime

    runtime = LinkRuntime(models_dir=REPO_MODELS, n_qubits_per_round=16,
                          enable_type_tagger=False)
    assert runtime.idle_pause_after_s == 0
    assert runtime.idle is False
    runtime.stop()


def test_start_demo_brings_up_a_fleet(client):
    """The console's empty state calls this; a hosted instance starts bare."""
    body = client.post("/api/start-demo?n=2").json()
    assert body["n_devices"] >= 2
    ids = {d["device_id"] for d in client.get("/api/state").json()["devices"]}
    assert {"plant-01", "plant-02"} <= ids
    # Idempotent: pressing it twice must not duplicate devices.
    again = client.post("/api/start-demo?n=2").json()
    assert again["created"] == []
