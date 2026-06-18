"""Aggregate task-level rank intervals into leaderboard-level intervals.

Each cell of the input is a rank interval ``(L, U)`` for one model on one task.
Union and quantile rules merge those intervals across tasks (columns).
"""

import pandas as pd


def merge_intervals_union(interval_collection: pd.DataFrame) -> pd.DataFrame:
    """Merge task intervals by enclosing union (widest interval per model).

    For each model (row), ``L`` is the minimum lower bound and ``U`` the maximum
    upper bound across tasks. This is the most conservative aggregation.

    Parameters
    ----------
    interval_collection : DataFrame
        Index = models, columns = tasks. Each cell is ``(L, U)``.

    Returns
    -------
    DataFrame
        Columns ``L``, ``U`` with one row per model.
    """
    lowers = interval_collection.map(lambda x: x[0])
    uppers = interval_collection.map(lambda x: x[1])

    merged_lowers = lowers.min(axis=1)
    merged_uppers = uppers.max(axis=1)

    return pd.concat([merged_lowers.rename('L'), merged_uppers.rename('U')], axis=1)


def merge_intervals_quantile(
    interval_collection: pd.DataFrame,
    alpha_ldb: float = 0.5,
) -> pd.DataFrame:
    """Merge task intervals by marginal quantiles with finite-sample correction.

    Lower and upper bounds are taken as quantiles across tasks, using
    ``q = alpha_ldb * N / (N + 1)`` with ``N`` the number of tasks.

    Parameters
    ----------
    interval_collection : DataFrame
        Index = models, columns = tasks. Each cell is ``(L, U)``.
    alpha_ldb : float
        Leaderboard error rate in ``[0, 1]`` (e.g. ``alpha_ldb``).

    Returns
    -------
    DataFrame
        Columns ``L``, ``U`` with one row per model.
    """
    if ((alpha_ldb > 1.0) or (alpha_ldb < 0.0)):
        raise ValueError(f'alpha_ldb must be between 0 and 1, but got {alpha_ldb}')

    lowers = interval_collection.map(lambda x: x[0])
    uppers = interval_collection.map(lambda x: x[1])
    N = interval_collection.shape[1]  # number of intermediate level nodes

    # Finite sample correction
    q = alpha_ldb * (N / (N + 1))
    merged_lowers = lowers.quantile(q / 2, axis=1, interpolation='lower')
    merged_uppers = uppers.quantile(1 - (q / 2), axis=1, interpolation='higher')

    return pd.concat([merged_lowers.rename('L'), merged_uppers.rename('U')], axis=1)
