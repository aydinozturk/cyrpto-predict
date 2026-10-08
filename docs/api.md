# Dashboard API

The dashboard service is started with `cryptopredict-web` and listens on
`CRYPTOPREDICT_HOST:CRYPTOPREDICT_PORT` (`0.0.0.0:8000` by default). All API
responses are JSON. When `CRYPTOPREDICT_DASHBOARD_TOKEN` is set, every
`/api/*` endpoint except `/api/health` requires
`Authorization: Bearer <token>`.

## Read endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Service status and package version |
| `GET` | `/api/config` | Models, intervals, defaults, paths, and auth status |
| `GET` | `/api/datasets` | Cached `<SYMBOL>_<interval>.csv` datasets |
| `GET` | `/api/ohlcv?symbol=BTCUSDT&interval=1h&limit=500` | Latest cached candles |
| `GET` | `/api/models` | Saved model metadata |
| `GET` | `/api/jobs` | Latest 100 jobs, newest first |
| `GET` | `/api/jobs/{id}` | One background job |

## Operations

`POST /api/fetch` accepts `symbol`, `interval`, `start`, and optional `end`.
`POST /api/train` accepts `symbol`, `interval`, optional `start`, `model`,
`horizon`, and optional artifact `name`. `POST /api/backtest` accepts the same
data/model fields plus `splits`, costs, thresholds, and `allow_short`; use model
`all` for a comparison table. These operations return HTTP 202 with a
`job_id`. Poll `/api/jobs/{id}` until its status is `done` or `failed`.

`POST /api/predict` is synchronous and accepts a saved artifact `name` plus an
optional `refresh` flag. `DELETE /api/models/{name}` deletes an artifact and
returns HTTP 204. Artifact names may contain only letters, digits, `_`, `.`,
and `-`.

The job queue is in memory and deliberately uses one worker. Queued jobs and
their status are lost when the process restarts; run Uvicorn with one worker.
