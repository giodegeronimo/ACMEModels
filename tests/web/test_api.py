from __future__ import annotations

import pytest


def test_list_models_returns_results(client):
    response = client.get("/api/models?limit=2")
    assert response.status_code == 200
    payload = response.get_json()
    assert "items" in payload
    assert len(payload["items"]) <= 2
    assert payload["total"] >= len(payload["items"])


def test_model_regex_validation(client):
    response = client.get("/api/models?q=*invalid[")
    assert response.status_code == 400
    payload = response.get_json()
    assert "invalid" in payload["error"].lower()


def test_ingest_rejects_low_scores(client):
    payload = {
        "name": "hf/some-model",
        "source_url": "https://huggingface.co/some-model",
        "submitted_by": "tester",
        "metrics": {
            "net_score": 0.6,
            "ramp_up_time": 0.3,
            "bus_factor": 0.7,
            "performance_claims": 0.7,
            "license": 0.8,
            "size_score": 0.6,
            "dataset_and_code_score": 0.5,
        },
    }
    response = client.post("/api/models/ingest", json=payload)
    assert response.status_code == 400
    message = response.get_json()["reason"]
    assert "ramp_up_time" in message


@pytest.mark.parametrize(
    "model_id",
    [
        "acme/solar-safeguard",
        "acme/solar-safeguard-edge",
    ],
)
def test_lineage_available(client, model_id):
    response = client.get(f"/api/models/{model_id}/lineage")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["root"] == model_id
    assert payload["nodes"]
    assert payload["edges"] is not None


def test_license_assessment_requires_fields(client):
    response = client.post("/api/license-check", json={})
    assert response.status_code == 400
    assert "required" in response.get_json()["error"]
