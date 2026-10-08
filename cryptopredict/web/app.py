"""FastAPI application serving the dashboard and the prediction pipeline."""

from __future__ import annotations

import hmac
import re
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cryptopredict import __version__, data, pipeline
from cryptopredict.models import available_models, load_model
from cryptopredict.web.jobs import JobRunner, json_safe
from cryptopredict.web.settings import SUPPORTED_INTERVALS, Settings

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class FetchRequest(BaseModel):
    symbol: str
    interval: str
    start: str
    end: str | None = None


class TrainRequest(BaseModel):
    symbol: str
    interval: str
    start: str | None = None
    model: str
    horizon: int = Field(default=1, ge=1)
    name: str | None = None


class PredictRequest(BaseModel):
    name: str
    refresh: bool = False


class BacktestRequest(BaseModel):
    symbol: str
    interval: str
    start: str | None = None
    model: str
    horizon: int = Field(default=1, ge=1)
    splits: int = Field(default=5, ge=2)
    fee_bps: float = Field(default=10.0, ge=0)
    slippage_bps: float = Field(default=0.0, ge=0)
    threshold: float = Field(default=0.0, ge=0)
    da_threshold: float = Field(default=0.0, ge=0)
    allow_short: bool = False


def _symbol(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized or not _SAFE_COMPONENT.fullmatch(normalized):
        raise HTTPException(status_code=400, detail="invalid symbol")
    return normalized


def _interval(value: str) -> str:
    normalized = value.strip()
    if normalized not in SUPPORTED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"unsupported interval {value!r}")
    return normalized


def _model_name(value: str) -> str:
    if not value or not _SAFE_MODEL_NAME.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail="invalid model name; use only letters, digits, underscore, dot and dash",
        )
    return value


def _registered_model(value: str, *, allow_all: bool = False) -> str:
    choices = [*available_models(), *(["all"] if allow_all else [])]
    if value not in choices:
        raise HTTPException(status_code=400, detail=f"unknown model {value!r}; available: {', '.join(choices)}")
    return value


def _cache_path(settings: Settings, symbol: str, interval: str) -> Path:
    return settings.data_dir / f"{symbol}_{interval}.csv"


def _artifact_path(settings: Settings, name: str) -> Path:
    return settings.model_dir / f"{_model_name(name)}.joblib"


def _load_cached(settings: Settings, symbol: str, interval: str) -> pd.DataFrame:
    path = _cache_path(settings, symbol, interval)
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found for {symbol} {interval}")
    return data.load_csv(path)


def _dataset_payload(symbol: str, interval: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "interval": interval,
        "rows": len(frame),
        "start": frame.index[0].isoformat() if len(frame) else None,
        "end": frame.index[-1].isoformat() if len(frame) else None,
    }


def _model_payload(name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "model": metadata.get("name"),
        "symbol": metadata.get("symbol"),
        "interval": metadata.get("interval"),
        "horizon": metadata.get("horizon"),
        "train_period": metadata.get("train_period"),
        "n_samples": metadata.get("n_samples"),
        "created_at": metadata.get("saved_at"),
    }


def _fetch(settings: Settings, symbol: str, interval: str, start: str | None, end: str | None = None) -> pd.DataFrame:
    return data.get_ohlcv(
        symbol,
        interval,
        start,
        end,
        cache_dir=settings.data_dir,
        base_url=settings.binance_url,
    )


def _frame_for_job(settings: Settings, symbol: str, interval: str, start: str | None) -> pd.DataFrame:
    if start is None:
        return _load_cached(settings, symbol, interval)
    return _fetch(settings, symbol, interval, start)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an application, optionally with explicit settings for tests/embedding."""
    settings = settings or Settings.from_env()
    runner = JobRunner()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.model_dir.mkdir(parents=True, exist_ok=True)
        yield
        runner.shutdown()

    application = FastAPI(title="cryptopredict dashboard", version=__version__, lifespan=lifespan)
    application.state.settings = settings
    application.state.jobs = runner

    @application.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        return JSONResponse(status_code=422, content={"detail": detail})

    @application.middleware("http")
    async def authenticate_api(request: Request, call_next):
        token = settings.dashboard_token
        if token and request.url.path.startswith("/api/") and request.url.path != "/api/health":
            expected = f"Bearer {token}"
            supplied = request.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, expected):
                return JSONResponse(status_code=401, content={"detail": "invalid or missing bearer token"})
        return await call_next(request)

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @application.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "version": __version__,
            "models": available_models(),
            "intervals": SUPPORTED_INTERVALS,
            "default_symbol": settings.default_symbol,
            "default_interval": settings.default_interval,
            "auth_required": settings.dashboard_token is not None,
            "data_dir": str(settings.data_dir),
            "model_dir": str(settings.model_dir),
        }

    @application.get("/api/datasets")
    def datasets() -> list[dict[str, Any]]:
        result = []
        if not settings.data_dir.exists():
            return result
        for path in sorted(settings.data_dir.glob("*.csv")):
            try:
                symbol, interval = path.stem.rsplit("_", 1)
                frame = data.load_csv(path)
            except (OSError, ValueError):
                continue
            result.append(_dataset_payload(symbol, interval, frame))
        return result

    @application.get("/api/ohlcv")
    def ohlcv(
        symbol: str = Query(default=settings.default_symbol),
        interval: str = Query(default=settings.default_interval),
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> dict[str, Any]:
        normalized_symbol = _symbol(symbol)
        normalized_interval = _interval(interval)
        try:
            frame = _load_cached(settings, normalized_symbol, normalized_interval).tail(limit)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise _http_error(exc) from exc
        bars = [
            {
                "t": timestamp.isoformat(),
                "o": float(row.open),
                "h": float(row.high),
                "l": float(row.low),
                "c": float(row.close),
                "v": float(row.volume),
            }
            for timestamp, row in frame.iterrows()
        ]
        return {"symbol": normalized_symbol, "interval": normalized_interval, "bars": bars}

    @application.post("/api/fetch", status_code=status.HTTP_202_ACCEPTED)
    def fetch(request: FetchRequest) -> dict[str, str]:
        symbol = _symbol(request.symbol)
        interval = _interval(request.interval)
        params = {"symbol": symbol, "interval": interval, "start": request.start, "end": request.end}

        def operation() -> dict[str, Any]:
            frame = _fetch(settings, symbol, interval, request.start, request.end)
            return _dataset_payload(symbol, interval, frame)

        return {"job_id": runner.submit("fetch", params, operation)}

    @application.post("/api/train", status_code=status.HTTP_202_ACCEPTED)
    def train(request: TrainRequest) -> dict[str, str]:
        symbol = _symbol(request.symbol)
        interval = _interval(request.interval)
        model_name = _registered_model(request.model)
        artifact_name = _model_name(
            request.name if request.name is not None else f"{symbol}_{interval}_{model_name}_h{request.horizon}"
        )
        if request.start is None and not _cache_path(settings, symbol, interval).is_file():
            raise HTTPException(status_code=400, detail=f"dataset not found for {symbol} {interval}; provide start")
        params = request.model_dump()
        params.update(symbol=symbol, interval=interval, model=model_name, name=artifact_name)

        def operation() -> dict[str, Any]:
            frame = _frame_for_job(settings, symbol, interval, request.start)
            model, info = pipeline.train(frame, model_name, horizon=request.horizon)
            path = _artifact_path(settings, artifact_name)
            metadata = pipeline.save_trained(
                model,
                info,
                path,
                symbol=symbol,
                interval=interval,
            )
            return _model_payload(artifact_name, metadata)

        return {"job_id": runner.submit("train", params, operation)}

    @application.get("/api/models")
    def models() -> list[dict[str, Any]]:
        result = []
        if not settings.model_dir.exists():
            return result
        for path in sorted(settings.model_dir.glob("*.joblib")):
            try:
                _, metadata = load_model(path)
            except (OSError, ValueError):
                continue
            result.append(_model_payload(path.stem, metadata))
        return sorted(result, key=lambda item: item["created_at"] or "", reverse=True)

    @application.delete("/api/models/{name}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_model(name: str) -> Response:
        path = _artifact_path(settings, name)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"model {name!r} not found")
        path.unlink()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.post("/api/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        path = _artifact_path(settings, request.name)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"model {request.name!r} not found")
        try:
            model, metadata = load_model(path)
            symbol = _symbol(str(metadata.get("symbol") or ""))
            interval = _interval(str(metadata.get("interval") or ""))
            frame = (
                _fetch(settings, symbol, interval, None)
                if request.refresh
                else _load_cached(settings, symbol, interval)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pipeline.OpenBarWarning)
                return json_safe(pipeline.predict_latest(model, metadata, frame))
        except HTTPException:
            raise
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise _http_error(exc) from exc

    @application.post("/api/backtest", status_code=status.HTTP_202_ACCEPTED)
    def backtest(request: BacktestRequest) -> dict[str, str]:
        symbol = _symbol(request.symbol)
        interval = _interval(request.interval)
        model_name = _registered_model(request.model, allow_all=True)
        if request.start is None and not _cache_path(settings, symbol, interval).is_file():
            raise HTTPException(status_code=400, detail=f"dataset not found for {symbol} {interval}; provide start")
        params = request.model_dump()
        params.update(symbol=symbol, interval=interval, model=model_name)

        def operation() -> dict[str, Any]:
            frame = _frame_for_job(settings, symbol, interval, request.start)
            options = dict(
                horizon=request.horizon,
                n_splits=request.splits,
                fee_bps=request.fee_bps,
                slippage_bps=request.slippage_bps,
                threshold=request.threshold,
                da_threshold=request.da_threshold,
                allow_short=request.allow_short,
                interval=interval,
            )
            if model_name == "all":
                table, _ = pipeline.compare_models(frame, **options)
                return {"table": table.reset_index().to_dict(orient="records")}
            report = pipeline.run_backtest(frame, model_name, **options)
            return {"summary": report.summary()}

        return {"job_id": runner.submit("backtest", params, operation)}

    @application.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return runner.list()

    @application.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        result = runner.get(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        return result

    static_dir = Path(__file__).with_name("static")
    static_dir.mkdir(parents=True, exist_ok=True)
    application.mount("/static", StaticFiles(directory=static_dir), name="static")

    @application.get("/", include_in_schema=False)
    def dashboard() -> Response:
        index = static_dir / "index.html"
        if index.is_file():
            return FileResponse(index)
        return HTMLResponse("<h1>cryptopredict dashboard</h1><p>Dashboard assets are not installed yet.</p>")

    return application


app = create_app()


def main() -> None:
    """Run the dashboard with the required single Uvicorn worker."""
    import uvicorn

    settings = Settings.from_env()
    uvicorn.run(
        "cryptopredict.web.app:app",
        host=settings.host,
        port=settings.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
