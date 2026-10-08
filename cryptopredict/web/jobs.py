"""Small in-memory, single-worker job queue used by the web API."""

from __future__ import annotations

import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable


def json_safe(value: Any) -> Any:
    """Return a JSON-safe copy, normalizing numpy values and non-finite floats."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobRunner:
    """Run submitted callables sequentially and retain their latest status."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cryptopredict-job")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, params: dict[str, Any], operation: Callable[[], Any]) -> str:
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "params": json_safe(params),
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, operation)
        return job_id

    def _run(self, job_id: str, operation: Callable[[], Any]) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["started_at"] = _now()
        try:
            result = json_safe(operation())
        except Exception as exc:  # Jobs expose failures through their status endpoint.
            with self._lock:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = str(exc)
                self._jobs[job_id]["finished_at"] = _now()
        else:
            with self._lock:
                self._jobs[job_id]["status"] = "done"
                self._jobs[job_id]["result"] = result
                self._jobs[job_id]["finished_at"] = _now()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())[-limit:]
            return [dict(job) for job in reversed(jobs)]

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
