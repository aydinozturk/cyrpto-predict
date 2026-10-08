import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cryptopredict import data
from cryptopredict.web.app import create_app
from cryptopredict.web.jobs import json_safe
from cryptopredict.web.settings import Settings

from .conftest import SAMPLE_CSV, load_sample_ohlcv


def _settings(tmp_path: Path, token: str | None = None) -> Settings:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copyfile(SAMPLE_CSV, data_dir / "BTCUSDT_1h.csv")
    return Settings(
        data_dir=data_dir,
        model_dir=tmp_path / "models",
        dashboard_token=token,
        default_symbol="BTCUSDT",
        default_interval="1h",
    )


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 20) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"done", "failed"}:
            return job
        time.sleep(0.02)
    pytest.fail(f"job {job_id} did not finish within {timeout}s")


def _submit(client: TestClient, path: str, payload: dict) -> dict:
    response = client.post(path, json=payload)
    assert response.status_code == 202, response.text
    job = _wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "done", job
    return job


def test_health_config_datasets_ohlcv_and_root(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/api/health").json() == {"status": "ok", "version": "0.1.0"}

        config = client.get("/api/config").json()
        assert config["default_symbol"] == "BTCUSDT"
        assert config["default_interval"] == "1h"
        assert config["auth_required"] is False
        assert {"ridge", "gbm", "zero"} <= set(config["models"])
        assert "1h" in config["intervals"]

        datasets = client.get("/api/datasets").json()
        assert datasets == [
            {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "rows": 600,
                "start": "2026-08-27T15:00:00+00:00",
                "end": "2026-09-21T14:00:00+00:00",
            }
        ]
        candles = client.get("/api/ohlcv", params={"symbol": "btcusdt", "interval": "1h", "limit": 2}).json()
        assert candles["symbol"] == "BTCUSDT" and len(candles["bars"]) == 2
        assert set(candles["bars"][0]) == {"t", "o", "h", "l", "c", "v"}
        assert client.get("/api/ohlcv", params={"symbol": "ETHUSDT", "interval": "1h"}).status_code == 404
        assert client.get("/").status_code == 200


def test_fetch_is_an_offline_background_job(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    calls = []

    def fake_get_ohlcv(symbol, interval, start=None, end=None, cache_dir=None, base_url=None):
        calls.append((symbol, interval, start, end, Path(cache_dir), base_url))
        frame = load_sample_ohlcv()
        data.save_csv(frame, Path(cache_dir) / f"{symbol}_{interval}.csv")
        return frame

    monkeypatch.setattr(data, "get_ohlcv", fake_get_ohlcv)
    with TestClient(create_app(settings)) as client:
        job = _submit(
            client,
            "/api/fetch",
            {"symbol": "ethusdt", "interval": "1h", "start": "2026-01-01"},
        )
        assert job["kind"] == "fetch"
        assert job["result"]["symbol"] == "ETHUSDT" and job["result"]["rows"] == 600
        assert calls == [("ETHUSDT", "1h", "2026-01-01", None, settings.data_dir, settings.binance_url)]


def test_train_predict_backtest_models_jobs_and_delete(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        trained = _submit(
            client,
            "/api/train",
            {"symbol": "BTCUSDT", "interval": "1h", "model": "ridge", "name": "btc-ridge"},
        )
        assert trained["kind"] == "train"
        assert trained["result"]["name"] == "btc-ridge"
        assert trained["result"]["model"] == "ridge"
        assert trained["result"]["n_samples"] > 0

        models = client.get("/api/models").json()
        assert len(models) == 1 and models[0] == trained["result"]

        response = client.post("/api/predict", json={"name": "btc-ridge"})
        assert response.status_code == 200, response.text
        prediction = response.json()
        assert prediction["model"] == "ridge" and prediction["symbol"] == "BTCUSDT"
        assert prediction["direction"] in {"up", "down", "flat"}
        assert prediction["target_time"] is not None

        single = _submit(
            client,
            "/api/backtest",
            {"symbol": "BTCUSDT", "interval": "1h", "model": "zero", "splits": 3},
        )
        assert {"model", "rmse", "total_return"} <= set(single["result"]["summary"])

        comparison = _submit(
            client,
            "/api/backtest",
            {"symbol": "BTCUSDT", "interval": "1h", "model": "all", "splits": 3},
        )
        assert len(comparison["result"]["table"]) == len(client.get("/api/config").json()["models"])
        assert json_safe({"metric": float("nan"), "nested": [float("inf")]}) == {
            "metric": None,
            "nested": [None],
        }

        jobs = client.get("/api/jobs").json()
        assert len(jobs) == 3
        assert jobs[0]["id"] == comparison["id"]
        assert client.get("/api/jobs/does-not-exist").status_code == 404

        response = client.delete("/api/models/btc-ridge")
        assert response.status_code == 204 and response.content == b""
        assert client.get("/api/models").json() == []
        assert client.post("/api/predict", json={"name": "btc-ridge"}).status_code == 404


def test_token_protects_all_api_routes_except_health(tmp_path):
    settings = _settings(tmp_path, token="correct horse")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/config").status_code == 401
        assert client.get("/api/config", headers={"Authorization": "Bearer wrong"}).status_code == 401
        authorized = {"Authorization": "Bearer correct horse"}
        assert client.get("/api/config", headers=authorized).status_code == 200
        assert client.get("/").status_code == 200


@pytest.mark.parametrize("name", ["../outside", "bad/name", "", "name with spaces"])
def test_train_rejects_unsafe_artifact_names(tmp_path, name):
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post(
            "/api/train",
            json={"symbol": "BTCUSDT", "interval": "1h", "model": "ridge", "name": name},
        )
        assert response.status_code in {400, 422}


def test_validation_errors_are_json(tmp_path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.post(
            "/api/train",
            json={"symbol": "BTCUSDT", "interval": "1h", "model": "unknown"},
        ).status_code == 400
        assert client.post(
            "/api/backtest",
            json={"symbol": "BTCUSDT", "interval": "bad", "model": "zero"},
        ).status_code == 400
        response = client.post("/api/predict", json={"name": "../outside"})
        assert response.status_code == 400 and "detail" in response.json()
        response = client.post("/api/train", json={})
        assert response.status_code == 422 and isinstance(response.json()["detail"], str)


def test_train_without_start_or_cache_is_rejected_before_queueing(tmp_path):
    settings = _settings(tmp_path)
    (settings.data_dir / "BTCUSDT_1h.csv").unlink()
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/train",
            json={"symbol": "BTCUSDT", "interval": "1h", "model": "ridge"},
        )
        assert response.status_code == 400
        assert client.get("/api/jobs").json() == []
