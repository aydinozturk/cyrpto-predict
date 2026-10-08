"""Time-ordered train/test splits for walk-forward validation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def walk_forward_splits(
    n: int,
    n_splits: int = 5,
    train_size: int | None = None,
    test_size: int | None = None,
    expanding: bool = True,
    gap: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` positional index arrays in time order.

    The ``n_splits`` test blocks are contiguous, non-overlapping and placed at the
    end of the series; every training index is strictly earlier than every test
    index of the same fold, with ``gap`` samples dropped in between.

    Parameters
    ----------
    n:
        Number of samples.
    n_splits:
        Number of folds (test blocks).
    train_size:
        Length of the first fold's training window. With ``expanding=False`` it
        is also the fixed (rolling) window length of every fold. If ``None`` it
        follows from ``n``, ``n_splits``, ``test_size`` and ``gap``.
    test_size:
        Length of each test block. If ``None``: ``(n - gap - train_size) //
        n_splits`` when ``train_size`` is given, else ``(n - gap) // (n_splits + 1)``.
    expanding:
        ``True``: training always starts at 0 (growing window). ``False``: a
        rolling window of ``train_size`` samples ending right before the gap.
    gap:
        Samples skipped between train and test. A target ``y[t] = log(close[t+h] /
        close[t])`` is only known at ``t + h``, so ``gap >= h - 1`` is required to
        keep training targets out of the test period; ``gap = h`` is the
        conservative choice.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if gap < 0:
        raise ValueError(f"gap must be >= 0, got {gap}")
    if train_size is not None and train_size < 1:
        raise ValueError(f"train_size must be >= 1, got {train_size}")

    if test_size is None:
        if train_size is not None:
            test_size = (n - gap - train_size) // n_splits
        else:
            test_size = (n - gap) // (n_splits + 1)
    if test_size < 1:
        raise ValueError(
            f"not enough samples (n={n}) for n_splits={n_splits}, gap={gap}, train_size={train_size}"
        )

    first_test_start = n - n_splits * test_size
    first_train_end = first_test_start - gap
    if train_size is None:
        train_size = first_train_end
    if first_train_end < train_size or train_size < 1:
        raise ValueError(
            f"not enough samples (n={n}) for train_size={train_size}, "
            f"n_splits={n_splits}, test_size={test_size}, gap={gap}"
        )

    for k in range(n_splits):
        test_start = first_test_start + k * test_size
        train_end = test_start - gap
        train_start = 0 if expanding else train_end - train_size
        yield (
            np.arange(train_start, train_end),
            np.arange(test_start, test_start + test_size),
        )
