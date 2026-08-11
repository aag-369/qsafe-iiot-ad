"""Smoke tests for the web dashboard's FastAPI backend. Uses TestClient
(in-process, no real HTTP server) so these run in the normal pytest suite
without needing a running uvicorn process."""

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from web.backend.app import app  # noqa: E402

client = fastapi_testclient.TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "liboqs_available" in body
    assert "kem_backend" in body


def test_project_info():
    r = client.get("/api/project-info")
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Q-Safe IIoT-AD"
    assert len(body["pipeline"]) == 4
    assert len(body["keywords"]) > 0


def test_results_summary():
    r = client.get("/api/results/summary")
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["train_metrics"]["f1"] <= 1.0
    assert "cpu_latency_reduction_pct" in body["benchmark"]


def test_probe():
    r = client.post(
        "/api/simulate/probe",
        json={"n_qubits": 32, "channel_error_prob": 0.02, "eve_intercept_prob": 0.3},
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["qber"] <= 1.0
    assert body["n_qubits"] == 32


def test_live_simulation_small():
    r = client.post(
        "/api/simulate/live",
        json={"n_rounds": 25, "n_qubits_per_round": 16, "inject_attack": True, "attack_intensity": 0.4},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["points"]) == 25
    assert all(p["profile"] in ("BIKE-L1", "HQC-128") for p in body["points"])


def test_frontend_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Q-Safe" in r.text
