# syntax=docker/dockerfile:1

# ---- build: produce a wheel for the package ----
FROM python:3.11-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build==1.2.2
COPY pyproject.toml README.md ./
COPY cryptopredict ./cryptopredict
RUN python -m build --wheel --outdir /dist

# ---- runtime: install the wheel with its extras, run as non-root ----
FROM python:3.11-slim AS runtime
# Optional dependency groups from pyproject.toml. The default image has the
# dashboard and LightGBM; add `deep` for the LSTM/GRU models (CPU-only PyTorch,
# a much larger image):
#   docker build --build-arg EXTRAS=web,ml,deep -t cryptopredict:deep .
ARG EXTRAS=web,ml
# Comma-separated, no spaces. torch is installed from TORCH_INDEX_URL first when
# EXTRAS contains `deep` (CPU wheels, no CUDA).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CRYPTOPREDICT_DATA_DIR=/data/cache \
    CRYPTOPREDICT_MODEL_DIR=/data/models \
    CRYPTOPREDICT_HOST=0.0.0.0 \
    CRYPTOPREDICT_PORT=8000

# LightGBM (`ml`) and PyTorch need the OpenMP runtime
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /dist /tmp/dist
# dependency ranges come from pyproject.toml via the wheel
# hadolint ignore=DL3013
RUN WHEEL="$(ls /tmp/dist/*.whl)" \
    && case ",${EXTRAS}," in \
         *,deep,*) pip install torch --index-url "${TORCH_INDEX_URL}" ;; \
       esac \
    && pip install "${WHEEL}${EXTRAS:+[${EXTRAS}]}" \
    && rm -rf /tmp/dist \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app \
    && mkdir -p /app /data/cache /data/models \
    && chown -R app:app /app /data

WORKDIR /app
USER app
VOLUME /data
EXPOSE 8000

# no curl in the slim image: probe the health endpoint with the stdlib
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('CRYPTOPREDICT_PORT', '8000'), timeout=4)"]

CMD ["cryptopredict-web"]
