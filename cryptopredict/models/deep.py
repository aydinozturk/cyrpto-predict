"""Optional CPU recurrent forecasters (LSTM, GRU) over windows of feature rows.

PyTorch is imported at module import time: the model registry loads this module
lazily, so installations without the ``deep`` extra keep every other model.
Training and inference run on the CPU so saved models stay portable.

Windows: the forecast for row ``t`` sees the scaled feature rows
``[t - lookback + 1, t]``.  A window never spans a gap in ``X``'s index (a step
longer than its most common step); training windows that would are dropped.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from cryptopredict.models.baseline import BaselineForecaster

SEED = 42


class _RecurrentRegressor(nn.Module):
    """A recurrent encoder whose last hidden state feeds a scalar regression head."""

    def __init__(
        self,
        kind: Literal["lstm", "gru"],
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        recurrent = nn.LSTM if kind == "lstm" else nn.GRU
        self.recurrent = recurrent(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.recurrent(values)
        return self.head(encoded[:, -1, :]).squeeze(-1)


def _modal_step(values: np.ndarray) -> Any | None:
    """Most common positive step of a sorted numeric index."""
    if len(values) < 2:
        return None
    deltas = np.diff(values)
    positive = deltas[deltas > 0]
    if not len(positive):
        return None
    steps, counts = np.unique(positive, return_counts=True)
    return steps[int(np.argmax(counts))]


def _follows(values: np.ndarray, step: Any | None) -> np.ndarray:
    """``follows[i]``: row ``i`` directly continues row ``i - 1`` (``follows[0]`` is True)."""
    follows = np.ones(len(values), dtype=bool)
    if len(values) >= 2 and step is not None:
        follows[1:] = np.diff(values) == step
    return follows


def _window_endpoints(follows: np.ndarray, lookback: int) -> np.ndarray:
    """Rows whose ``lookback`` window lies inside one contiguous run."""
    run_length = np.zeros(len(follows), dtype=np.int64)
    for position, continues in enumerate(follows):
        run_length[position] = run_length[position - 1] + 1 if position and continues else 1
    return np.flatnonzero(run_length >= lookback)


class _SequenceForecaster(BaselineForecaster):
    """Shared estimator behind :class:`LSTMForecaster` and :class:`GRUForecaster`.

    Training: features are standardized and the target is z-scored with
    statistics of the training rows only.  The chronologically last
    ``validation_fraction`` of the windows is held out for early stopping (Huber
    loss, AdamW); training stops after ``patience`` epochs without improvement,
    ``max_epochs`` or ``max_train_seconds``, and keeps the best epoch's weights.

    ``predict(X)`` returns one value per row of ``X`` and only looks back.  The
    first ``lookback - 1`` rows lack a full window inside ``X``; they are
    completed with the last ``lookback - 1`` training rows when ``X`` starts after
    the training data and within ``lookback`` steps of it (as in walk-forward
    folds; the skipped rows of a fold gap are simply absent from the window).
    Otherwise (and after a gap inside ``X``) the window is left-padded by
    repeating its first row.  Rows ``lookback - 1`` and later of a contiguous
    ``X`` use windows made of ``X`` alone, so passing the last ``lookback`` rows
    is enough for a live forecast.
    """

    heavy = True
    _kind: Literal["lstm", "gru"]

    def __init__(
        self,
        lookback: int = 48,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        validation_fraction: float = 0.15,
        max_train_seconds: float = 60.0,
        random_state: int = SEED,
    ) -> None:
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.max_train_seconds = max_train_seconds
        self.random_state = random_state

    def _validate_hyperparameters(self) -> None:
        for name in ("lookback", "hidden_size", "num_layers", "batch_size", "max_epochs", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout!r}")
        if not self.learning_rate > 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if not self.weight_decay >= 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay!r}")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(f"validation_fraction must be in (0, 1), got {self.validation_fraction!r}")
        if not self.max_train_seconds > 0.0:
            raise ValueError(f"max_train_seconds must be > 0, got {self.max_train_seconds!r}")

    @staticmethod
    def _index_values(X: pd.DataFrame) -> np.ndarray:
        """The index as plain numbers; datetimes become int64 nanoseconds (any unit or tz)."""
        index = X.index
        if not index.is_monotonic_increasing or index.has_duplicates:
            raise ValueError("X index must be strictly increasing for sequence models")
        if isinstance(index, pd.DatetimeIndex):
            return index.as_unit("ns").asi8
        values = index.to_numpy()
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"X index must be datetime or numeric for sequence models, got {values.dtype}")
        return values

    def _network(self, n_features: int) -> _RecurrentRegressor:
        # Seed the weight initialization without touching the global RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            return _RecurrentRegressor(
                self._kind, n_features, int(self.hidden_size), int(self.num_layers), float(self.dropout)
            ).cpu()

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._validate_hyperparameters()
        lookback = int(self.lookback)
        if len(X) < lookback:
            raise ValueError(f"{self.name} needs at least lookback={lookback} training rows, got {len(X)}")
        raw = X.to_numpy(dtype="float64")
        if not np.isfinite(raw).all():
            raise ValueError(f"{self.name}: X contains NaN or infinite values")
        index = self._index_values(X)
        self.index_step_ = _modal_step(index)
        follows = _follows(index, self.index_step_)
        endpoints = _window_endpoints(follows, lookback)
        if not len(endpoints):
            raise ValueError(f"{self.name}: no gap-free window of lookback={lookback} rows in the training data")

        self.scaler_ = StandardScaler().fit(raw)
        scaled = self.scaler_.transform(raw).astype("float32")
        self.target_mean_ = float(y.mean())
        spread = float(y.std())
        self.target_scale_ = spread if spread > np.finfo("float64").eps else 1.0
        targets = ((y - self.target_mean_) / self.target_scale_).astype("float32")
        windows = np.stack([scaled[end - lookback + 1 : end + 1] for end in endpoints])
        targets = targets[endpoints]

        n_validation = int(np.ceil(len(windows) * self.validation_fraction)) if len(windows) > 1 else 0
        split = len(windows) - n_validation
        train_x, train_y = torch.from_numpy(windows[:split]), torch.from_numpy(targets[:split])
        val_x, val_y = torch.from_numpy(windows[split:]), torch.from_numpy(targets[split:])
        monitor_x, monitor_y = (val_x, val_y) if n_validation else (train_x, train_y)

        network = self._network(raw.shape[1])
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay)
        )
        loss_function = nn.HuberLoss()
        generator = torch.Generator().manual_seed(int(self.random_state))
        best_loss, best_state, stale = float("inf"), copy.deepcopy(network.state_dict()), 0
        started = time.monotonic()
        out_of_time = False
        self.n_epochs_ = 0
        for epoch in range(int(self.max_epochs)):
            network.train()
            order = torch.randperm(len(train_x), generator=generator)
            for offset in range(0, len(order), int(self.batch_size)):
                batch = order[offset : offset + int(self.batch_size)]
                optimizer.zero_grad(set_to_none=True)
                loss_function(network(train_x[batch]), train_y[batch]).backward()
                optimizer.step()
                if time.monotonic() - started >= self.max_train_seconds:
                    out_of_time = True
                    break
            network.eval()
            with torch.no_grad():
                monitored = float(loss_function(network(monitor_x), monitor_y))
            self.n_epochs_ = epoch + 1
            if monitored < best_loss - 1e-8:
                best_loss, best_state, stale = monitored, copy.deepcopy(network.state_dict()), 0
            else:
                stale += 1
            if stale >= self.patience or out_of_time:
                break

        network.load_state_dict(best_state)
        network.eval()
        self.network_ = network
        self.best_validation_loss_ = best_loss
        self.fit_seconds_ = time.monotonic() - started
        self.n_features_in_ = raw.shape[1]
        self.stopped_by_time_ = out_of_time
        # History for the first prediction windows: the training data's final contiguous rows.
        run_start = int(np.flatnonzero(~follows)[-1]) if (~follows).any() else 0
        tail_start = max(run_start, len(raw) - (lookback - 1))
        self.history_ = scaled[tail_start:]
        self.history_end_ = index[-1]

    def _uses_history(self, index: np.ndarray) -> bool:
        if not len(self.history_) or self.index_step_ is None:
            return False
        distance = index[0] - self.history_end_
        return bool(0 < distance <= self.index_step_ * int(self.lookback))

    def _windows(self, X: pd.DataFrame) -> np.ndarray:
        lookback = int(self.lookback)
        index = self._index_values(X)
        follows = _follows(index, self.index_step_)
        scaled = self.scaler_.transform(X[self.feature_names_].to_numpy(dtype="float64")).astype("float32")
        windows = np.empty((len(scaled), lookback, scaled.shape[1]), dtype="float32")
        context = list(self.history_) if self._uses_history(index) else []
        for position, row in enumerate(scaled):
            if position and not follows[position]:
                context = []
            context.append(row)
            context = context[-lookback:]
            padding = [context[0]] * (lookback - len(context))
            windows[position] = np.stack(padding + context)
        return windows

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "network_"):
            raise ValueError(f"{self.name} is not fitted")
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}")
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"{self.name}: X is missing training feature(s) {missing}")
        if len(X) == 0:
            return np.empty(0, dtype="float64")
        if not np.isfinite(X[self.feature_names_].to_numpy(dtype="float64")).all():
            raise ValueError(f"{self.name}: X contains NaN or infinite values")
        windows = torch.from_numpy(self._windows(X))
        with torch.no_grad():
            scaled = torch.cat(
                [self.network_(windows[i : i + 1024]) for i in range(0, len(windows), 1024)]
            ).numpy()
        return scaled.astype("float64") * self.target_scale_ + self.target_mean_

    def __getstate__(self) -> dict[str, Any]:
        """Pickle the fitted weights as CPU tensors instead of a live module."""
        state = dict(super().__getstate__())  # sklearn may return the live __dict__
        network = state.pop("network_", None)
        if network is not None:
            state["network_state_"] = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state = dict(state)
        weights = state.pop("network_state_", None)
        super().__setstate__(state)
        if weights is not None:
            network = self._network(self.n_features_in_)
            network.load_state_dict(weights)
            network.eval()
            self.network_ = network


class LSTMForecaster(_SequenceForecaster):
    """Long short-term memory network over the last ``lookback`` feature rows."""

    name = "lstm"
    _kind = "lstm"


class GRUForecaster(_SequenceForecaster):
    """Gated recurrent unit network over the last ``lookback`` feature rows."""

    name = "gru"
    _kind = "gru"
