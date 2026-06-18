"""Metrics and helpers for rank intervals and coverage evaluation."""

import numpy as np
import pandas as pd


def L_U_to_CI(ranking: pd.DataFrame) -> pd.Series:
    """Convert ``L`` / ``U`` columns to a Series of ``(L, U)`` tuples.

    Parameters
    ----------
    ranking : DataFrame
        Must have columns ``L`` and ``U``.

    Returns
    -------
    Series
        One ``(L, U)`` tuple per row index label.
    """
    return ranking.apply(lambda x: tuple(x), axis=1)


def get_top_k(set_ranks: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return models that can appear in the top ``k`` ranks.

    A model is included when its upper bound ``U`` is at least
    ``highest_rank - k + 1`` (rank 1 = worst).

    Parameters
    ----------
    set_ranks : DataFrame
        Columns ``L``, ``U`` indexed by model.
    k : int
        Top-k cutoff.

    Returns
    -------
    DataFrame
        Subset of ``set_ranks``, sorted by ``L`` descending.
    """
    highest_rank = set_ranks['U'].max()
    k_rank = highest_rank - k + 1
    return set_ranks[set_ranks['U'] >= k_rank].sort_values(by='L', ascending=False)


def calc_interval_width(uppers: np.ndarray, lowers: np.ndarray) -> float:
    """Normalized average rank interval width across models.

    Parameters
    ----------
    uppers, lowers : array-like
        Upper and lower rank bounds per model.

    Returns
    -------
    float
        Normalized average rank interval width. Values are in [0, 1].
    """
    M = np.size(uppers)
    return (1 / (M * (M - 1))) * np.sum(uppers - lowers).item()


def calc_true_rank(true_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Partial rank sets from noise-free scores (rank 1 = worst / lowest score).

    Parameters
    ----------
    true_scores : array-like
        Length ``M``; higher values indicate better performance.

    Returns
    -------
    lowers, uppers : ndarray
        Length ``M``; each model's true rank interval when ties are possible.
    """
    M = np.size(true_scores)
    lowers = np.full(M, 1)
    uppers = np.full(M, M)
    for j in range(M):
        lowers[j] += np.sum(true_scores[j] > np.delete(true_scores, j)).item()
        uppers[j] -= np.sum(true_scores[j] < np.delete(true_scores, j)).item()
    return lowers, uppers


def calc_coverage(
    task_intervals: pd.DataFrame,
    true_ranks: pd.Series,
) -> tuple[float, bool]:
    """Fraction of models whose interval contains the true rank set.

    Parameters
    ----------
    task_intervals : DataFrame
        Columns ``L``, ``U`` per model.
    true_ranks : Series
        Each entry is ``(true_L, true_U)`` for that model.

    Returns
    -------
    coverage : float
        Proportion of models covered.
    all_covered : bool
        True if all models are covered.
    """
    lowers = task_intervals['L']
    uppers = task_intervals['U']
    true_lowers = true_ranks.apply(lambda x: x[0])
    true_uppers = true_ranks.apply(lambda x: x[1])
    coverage = np.size(np.where((lowers <= true_lowers) & (true_uppers <= uppers))[0]) / np.size(true_lowers)
    return coverage, coverage == 1


def calc_coverage_per_model(
    leaderboard_intervals: pd.DataFrame,
    task_intervals: pd.DataFrame,
) -> pd.Series:
    """Per-model fraction of tasks whose interval lies inside the leaderboard interval.

    Parameters
    ----------
    leaderboard_intervals : DataFrame
        Columns ``L``, ``U``; one row per model (aggregated leaderboard bounds).
    task_intervals : DataFrame
        Index = models, columns = tasks. Each cell is ``(L, U)`` for that
        model on that task (observed intervals or true rank sets).

    Returns
    -------
    Series
        For each model, the share of tasks with
        ``leaderboard_L <= task_L`` and ``task_U <= leaderboard_U``.
    """
    count_datasets = {}
    for j, model_interval in leaderboard_intervals.iterrows():
        L_j = model_interval['L']
        U_j = model_interval['U']
        count_datasets[j] = 0
        n_datasets = task_intervals.shape[1]
        for b in task_intervals.columns:
            low_jb = task_intervals.loc[j, b][0]
            up_jb = task_intervals.loc[j, b][1]
            if ((L_j <= low_jb) and (U_j >= up_jb)):
                count_datasets[j] += 1
        count_datasets[j] = count_datasets[j] / n_datasets
    return pd.Series(count_datasets)


def summarize_coverage_per_model(
    coverage_per_model: pd.Series,
    alpha: float,
) -> tuple[bool, float]:
    """Summarize per-model task coverage against a nominal level.

    Parameters
    ----------
    coverage_per_model : Series
        Output of ``calc_coverage_per_model``.
    alpha : float
        Nominal error rate; checks coverage against ``1 - alpha``.

    Returns
    -------
    all_covered : bool
        True if every model meets ``coverage >= 1 - alpha``.
    average_covered : float
        Mean coverage across models.
    """
    all_covered = all(coverage_per_model >= 1 - alpha)
    average_covered = coverage_per_model.mean()
    return all_covered, average_covered
