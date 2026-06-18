"""Task-level ranking from pairwise comparisons and bootstrap scores.

Build rank intervals ``[L, U]`` for each model (rank 1 = best) from:
paired t-tests or Wilcoxon tests, multiple-comparison correction, and optional
bootstrap resampling.
"""

import pandas as pd
import numpy as np
from itertools import combinations, permutations
from scipy.stats import wilcoxon, t
from statsmodels.stats.multitest import multipletests

_WILCOXON_KWARGS = {
    'zero_method': 'pratt',
    'correction': True,
    'method': 'approx',
}

# --- Data preparation ---
def pivot_scores_by_model(
    df_long: pd.DataFrame,
    index_name: str,
    col_name: str,
    val_name: str,
) -> pd.DataFrame:
    """Pivot long-form scores to a wide matrix for paired comparisons.

    Parameters
    ----------
    df_long : DataFrame
        Long-format table with one score per row.
    index_name : str
        Column identifying paired observations (e.g. fold or question id).
    col_name : str
        Column identifying models.
    val_name : str
        Column with numeric scores.

    Returns
    -------
    DataFrame
        Shape ``(n_observations, n_models)``; rows are paired observations,
        columns are models.
    """
    pivot_data = df_long.pivot(index=index_name, columns=col_name, values=val_name)
    pivot_data = pivot_data.reset_index(drop=True)
    pivot_data.index.name = None
    pivot_data.columns.name = None
    return pivot_data


# --- Pairwise comparisons ---
def _validate_base_values(data=None, means=None, cov_matrix=None, n=None):
    """Normalize inputs for paired tests (raw data or summary statistics)."""
    if data is not None:
        if isinstance(data, pd.DataFrame):
            out = data
        elif isinstance(data, np.ndarray):
            out = pd.DataFrame(data, columns=range(data.shape[1]))
        else:
            raise ValueError("Error 'data' must be a DataFrame or a numpy array")
        if out.shape[0] < 1 or out.shape[1] < 1:
            raise ValueError(
                "'data' must have at least one row (observation) and one column (model)."
            )
        return out
    
    elif ((means is not None) and (cov_matrix is not None) and (n is not None)):    
        if not isinstance(n, int):
            raise ValueError("'n' must be an integer. Got n: %s." % type(n))

        is_means_series = isinstance(means, pd.Series)
        is_cov_dataframe = isinstance(cov_matrix, pd.DataFrame)
        if is_means_series and is_cov_dataframe: # means is a series, cov_matrix is a dataframe
            return means, cov_matrix
        if is_means_series: # means is a series, cov_matrix is not a dataframe
            cov_matrix = pd.DataFrame(cov_matrix, columns=means.index, index=means.index)
            return means, cov_matrix
        elif is_cov_dataframe: # means is not a series, cov_matrix is a dataframe
            means = pd.Series(means, index=cov_matrix.columns)
            return means, cov_matrix
        elif ((isinstance(means, np.ndarray)) and (isinstance(cov_matrix, np.ndarray))): # means and cov_matrix are numpy arrays
            p_list = range(means.shape[0])
            means = pd.Series(means, index=p_list)
            cov_matrix = pd.DataFrame(cov_matrix, columns=p_list, index=p_list)
            return means, cov_matrix
        else:
            raise ValueError("Error 'means' must be a Series or a numpy array, \
            and 'cov_matrix' must be a DataFrame or a numpy array. \
            Got means: %s, cov_matrix: %s." % (type(means), type(cov_matrix)
        ))
    
    else:
        raise ValueError('Error - wrong input. \
        Got means: %s, cov_matrix: %s, n: %s.' % (type(means), type(cov_matrix), type(n)))


def _paired_ttest_from_summary(means_arr, cov_arr, n, pairs_idx, alternative='less'):
    """Vectorized paired t-test from means, covariance, and sample size.

    Parameters
    ----------
    means_arr, cov_arr : ndarray
        Model means and covariance matrix (aligned by column order).
    n : int
        Number of observations used to estimate the covariance.
    pairs_idx : list of tuple[int, int]
        Unordered model index pairs ``(i, j)``.
    alternative : str
        One-sided or two-sided alternative (see ``calc_paired_tests``).

    Returns
    -------
    diff_mean, t_stat, p_value : ndarray
        One value per pair in ``pairs_idx``.
    """
    i_vals = np.array([pi[0] for pi in pairs_idx])
    j_vals = np.array([pi[1] for pi in pairs_idx])
    mean1 = means_arr[i_vals]
    mean2 = means_arr[j_vals]
    diff_mean = mean1 - mean2
    var1 = cov_arr[i_vals, i_vals]
    var2 = cov_arr[j_vals, j_vals]
    cov12 = cov_arr[i_vals, j_vals]
    se_diff = np.sqrt((var1 + var2 - 2 * cov12) / n)
    df = n - 1
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = np.where(np.isclose(se_diff, 0), np.nan, diff_mean / se_diff)
    if alternative == 'two-sided':
        p_value = np.where(np.isclose(se_diff, 0), np.nan, 2 * t.sf(np.abs(t_stat), df))
    elif alternative == 'greater':
        p_value = np.where(np.isclose(se_diff, 0), np.nan, t.sf(t_stat, df))
    elif alternative == 'less':
        p_value = np.where(np.isclose(se_diff, 0), np.nan, t.cdf(t_stat, df))
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return diff_mean, t_stat, p_value


def _mirror_paired_test_results(res_df, alternative='less'):
    """Expand unordered-pair t-test results to both directed comparisons.

    Valid for the paired t-test only: for ``diff = score_1 - score_2``, the
    reverse pair flips ``diff_mean`` and ``statistic`` and complements one-sided
    p-values. Not used for Wilcoxon (signed ranks are not mirrorable).
    """
    statistic = pd.to_numeric(res_df['statistic'], errors='coerce')
    pvalue = pd.to_numeric(res_df['pvalue'], errors='coerce')
    if alternative == 'two-sided':
        mirror_pvalue = pvalue.to_numpy()
    else:
        mirror_pvalue = 1 - pvalue.to_numpy()
    mirrored = pd.DataFrame({
        'candidate_1': res_df['candidate_2'].to_numpy(),
        'candidate_2': res_df['candidate_1'].to_numpy(),
        'diff_mean': -res_df['diff_mean'].to_numpy(dtype=np.float64),
        'statistic': -statistic.to_numpy(),
        'pvalue': mirror_pvalue,
    })
    return pd.concat([res_df, mirrored], ignore_index=True)


def calc_paired_tests(
    base_values: dict,
    from_summary: bool = False,
    test_func_name: str = 'ttest',
    alternative: str = 'less',
) -> pd.DataFrame:
    """Run all pairwise model comparisons on paired observations or summaries.

    For each directed pair ``(candidate_1, candidate_2)`` the test uses
    per-observation differences ``score_1 - score_2``. A small p-value indicates
    evidence against the null in the direction given by ``alternative`` (scipy
    convention). Choose ``alternative`` so that rejection corresponds to
    ``candidate_1`` ranking better than ``candidate_2``, given whether higher
    or lower raw scores mean better performance.

    Parameters
    ----------
    base_values : dict
        If ``from_summary=False``: ``{'data': DataFrame or ndarray}`` with shape
        ``(n_obs, n_models)``.
        If ``from_summary=True``: ``{'means': ..., 'cov_matrix': ..., 'n': int}``.
    from_summary : bool
        If True, run a paired t-test from summary statistics only (no Wilcoxon).
    test_func_name : {'ttest', 'wilcoxon'}
        Test to apply. Wilcoxon requires raw observations.
    alternative : {'less', 'greater', 'two-sided'}
        Alternative hypothesis passed to scipy.

    Returns
    -------
    DataFrame
        Columns ``candidate_1``, ``candidate_2``, ``diff_mean``, ``statistic``,
        ``pvalue``. One row per directed pair; ``p * (p - 1)`` rows for ``p``
        models.

    Notes
    -----
    - **t-test**: unordered pairs are computed, then mirrored to both directions.
    - **Wilcoxon**: every directed pair is tested explicitly (no mirroring).
    - **Wilcoxon settings**: ``zero_method='pratt'``, ``correction=True``,
      ``method='approx'`` (see ``_WILCOXON_KWARGS``).
    """
    supported_paired_tests = {'ttest', 'wilcoxon'}
    if test_func_name not in supported_paired_tests:
        raise ValueError('test_func_name must be one of: %s' % ', '.join(supported_paired_tests))

    if from_summary:
        if test_func_name != 'ttest':
            raise ValueError(
                "from_summary only supports test_func_name='ttest'. "
                'Wilcoxon requires raw paired observations.'
            )
        means = base_values['means']
        cov_matrix = base_values['cov_matrix']
        n = base_values['n']
        means, cov_matrix = _validate_base_values(means=means, cov_matrix=cov_matrix, n=n)
        p = means.shape[0]
        if (cov_matrix.shape[0] != p) or (cov_matrix.shape[1] != p):
            raise ValueError('means and cov_matrix are not aligned.')
        p_list = list(cov_matrix.columns)
        means_arr = means.loc[p_list].values
        cov_arr = cov_matrix.loc[p_list, p_list].values
        # t-test: mirror to get both directions
        pairs_idx = list(combinations(range(p), 2))  
        pairs_names = [(p_list[i], p_list[j]) for i, j in pairs_idx]
        diff_mean, t_stat, p_value = _paired_ttest_from_summary(
            means_arr, cov_arr, n, pairs_idx, alternative=alternative
        )
        res_df = pd.DataFrame({
            'candidate_1': [p[0] for p in pairs_names],
            'candidate_2': [p[1] for p in pairs_names],
            'diff_mean': diff_mean,
            'statistic': t_stat,
            'pvalue': p_value,
        })
        res_df = _mirror_paired_test_results(res_df, alternative=alternative)
        res_df['statistic'] = res_df['statistic'].replace({np.nan: None})
        res_df['pvalue'] = res_df['pvalue'].replace({np.nan: None})
        return res_df

    # From observations
    data = _validate_base_values(data=base_values['data'])
    p_list = list(data.columns)
    arr = data.values.astype(np.float64)
    n_obs, p = arr.shape
    # t-test: (j,i) results follow from (i,j) by sign flip; Wilcoxon must be run per direction.
    if test_func_name == 'ttest':
        pairs_idx = list(combinations(range(p), 2))
    else:
        pairs_idx = list(permutations(range(p), 2))
    i_vals = np.array([pi[0] for pi in pairs_idx])
    j_vals = np.array([pi[1] for pi in pairs_idx])
    pair_diffs = arr[:, i_vals] - arr[:, j_vals]  # (n_obs, n_pairs)
    diff_means = np.mean(pair_diffs, axis=0)
    min_d = np.min(pair_diffs, axis=0)
    max_d = np.max(pair_diffs, axis=0)
    constant = min_d == max_d

    if test_func_name == 'ttest':
        std_d = np.std(pair_diffs, axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stat = np.where(constant, 0, diff_means / (std_d / np.sqrt(n_obs)))
        df = n_obs - 1
        if alternative == 'two-sided':
            p_val = np.where(constant, 1.0, 2 * t.sf(np.abs(t_stat), df))
        elif alternative == 'greater':
            p_val = np.where(constant, 1.0, t.sf(t_stat, df))
        else:  # 'less'
            p_val = np.where(constant, 1.0, t.cdf(t_stat, df))
        res_df = pd.DataFrame({
            'candidate_1': [p_list[i] for i, _ in pairs_idx],
            'candidate_2': [p_list[j] for _, j in pairs_idx],
            'diff_mean': diff_means,
            'statistic': np.where(constant, 0, t_stat),
            'pvalue': p_val,
        })
        return _mirror_paired_test_results(res_df, alternative=alternative)

    # wilcoxon: no vectorized API; test each directed pair explicitly (no mirroring)
    rows = []
    for idx, (i, j) in enumerate(pairs_idx):
        pair_diff = pair_diffs[:, idx]
        pair_diff_mean = diff_means[idx]
        if constant[idx]:
            rows.append((p_list[i], p_list[j], pair_diff_mean, 0, 1.0))
        else:
            wres = wilcoxon(pair_diff, alternative=alternative, **_WILCOXON_KWARGS)
            rows.append((p_list[i], p_list[j], pair_diff_mean, wres.statistic, wres.pvalue))
    return pd.DataFrame(rows, columns=['candidate_1', 'candidate_2', 'diff_mean', 'statistic', 'pvalue'])

# --- Multiple-comparison correction ---
def multipletests_correction(
    paired_test_res: pd.DataFrame,
    alpha: float = 0.1,
    correction_method: str = 'holm',
) -> np.ndarray:
    """Apply a multiple-comparison procedure to all pairwise p-values.

    Parameters
    ----------
    paired_test_res : DataFrame
        Output of ``calc_paired_tests``; must contain a ``pvalue`` column.
    alpha : float
        Family-wise or FDR error rate.
    correction_method : str
        Method passed to ``statsmodels.stats.multitest.multipletests``
        (e.g. ``'holm'``, ``'fdr_bh'``).

    Returns
    -------
    ndarray of bool
        ``True`` where the null is rejected after correction.
    """
    multi_res = multipletests(paired_test_res['pvalue'], alpha, method=correction_method)
    return multi_res[0]


# --- Ranking from paired tests ---
def calc_set_ranks(
    multi_test_res: pd.DataFrame,
    c1_col: str = 'candidate_1',
    c2_col: str = 'candidate_2',
) -> pd.DataFrame:
    """Derive rank intervals from a table of corrected rejections.

    If directed pair ``(j, k)`` has ``reject=True``, treat ``j`` as ranking
    better than ``k`` (smaller rank number) and tighten ``L``, ``U`` accordingly.

    Parameters
    ----------
    multi_test_res : DataFrame
        Must contain ``reject`` plus candidate columns.
    c1_col, c2_col : str
        Column names for the directed pair.

    Returns
    -------
    DataFrame
        Columns ``L``, ``U`` indexed by candidate (initially ``[1, p]``).
    """
    candidate_list = multi_test_res[c1_col].unique()
    p = len(candidate_list)
    # Init to [1, p] for all candidates
    set_ranks = pd.DataFrame({'L': np.full(p, 1), 'U': np.full(p, p)}, index=candidate_list)
    for pair in combinations(candidate_list, 2):
        j = pair[0]
        k = pair[1]
        # check if the pair j, k exist:
        jk_reject = False
        kj_reject = False
        test_res_jk = multi_test_res[(multi_test_res[c1_col] == j) & (multi_test_res[c2_col] == k)]
        if not test_res_jk.empty:
            jk_reject = test_res_jk['reject'].values[0]
        test_res_kj = multi_test_res[(multi_test_res[c1_col] == k) & (multi_test_res[c2_col] == j)]
        if not test_res_kj.empty:
            kj_reject = test_res_kj['reject'].values[0]
        if not (jk_reject or kj_reject): # j = k
            continue
        elif jk_reject: # j < k
            set_ranks.loc[j, 'U'] -= 1
            set_ranks.loc[k, 'L'] += 1
        else: # k < j
            set_ranks.loc[k, 'U'] -= 1
            set_ranks.loc[j, 'L'] += 1
    return set_ranks


def calc_simultaneous_intervals(
    paired_test_res: pd.DataFrame,
    alpha: float = 0.1,
    c1_col: str = 'candidate_1',
    c2_col: str = 'candidate_2',
    correction_method: str = 'holm',
) -> pd.DataFrame:
    """Rank intervals with one global multiple-comparison correction.

    Holm (or another method) is applied once to all ``p * (p - 1)`` directed
    p-values. Rejections are aggregated per candidate: outgoing wins tighten
    ``U``, incoming losses tighten ``L``.

    Parameters
    ----------
    paired_test_res : DataFrame
        Output of ``calc_paired_tests``.
    alpha : float
        Significance level for the global correction.
    c1_col, c2_col : str
        Directed pair column names.
    correction_method : str
        Passed to ``multipletests``.

    Returns
    -------
    DataFrame
        Columns ``L``, ``U`` per candidate; ``1 <= L <= U <= p``.
    """
    candidate_list = paired_test_res[c1_col].unique()
    n_candidates = len(candidate_list)
    corrected_rejects = multipletests_correction(paired_test_res, alpha=alpha, correction_method=correction_method)

    reject_matrix = (
        paired_test_res.assign(reject=corrected_rejects)
        .pivot(index=c1_col, columns=c2_col, values='reject')
        .reindex(index=candidate_list, columns=candidate_list, fill_value=False)
        .rename_axis(index=None, columns=None)
    )
    reject_matrix = reject_matrix.fillna(False).astype(bool)

    # For each candidate c:
    # - Row c counts rejections of (c, other): c < other  => contributes to U
    # - Col c counts rejections of (other, c): other < c => contributes to L
    wins_count = reject_matrix.sum(axis=1).to_numpy()
    losses_count = reject_matrix.sum(axis=0).to_numpy()

    set_ranks = pd.DataFrame(
        {
            'L': 1 + losses_count,
            'U': n_candidates - wins_count,
        },
        index=candidate_list,
    )
    return set_ranks


def calc_marginal_intervals(
    paired_test_res: pd.DataFrame,
    alpha: float = 0.1,
    c1_col: str = 'candidate_1',
    c2_col: str = 'candidate_2',
    correction_method: str = 'holm',
) -> pd.DataFrame:
    """Rank intervals with separate multiple-comparison correction per candidate.

    For each candidate, Holm correction is applied at level ``alpha / 2`` to
    outgoing comparisons (row of the p-value matrix) and separately to incoming
    comparisons (column). This is the default in ``ranking_from_multiple_paired_tests``.

    Parameters
    ----------
    paired_test_res : DataFrame
        Output of ``calc_paired_tests``.
    alpha : float
        Nominal level; each marginal family uses ``alpha / 2``.
    c1_col, c2_col : str
        Directed pair column names.
    correction_method : str
        Passed to ``multipletests``.

    Returns
    -------
    DataFrame
        Columns ``L``, ``U`` per candidate; ``1 <= L <= U <= p``.
    """
    candidate_list = paired_test_res[c1_col].unique()
    n_candidates = len(candidate_list)
    all_ranks = {}
    pvalue_matrix = paired_test_res.pivot(index=c1_col, 
                                          columns=c2_col, 
                                          values='pvalue').rename_axis(index=None, columns=None)
    for c in candidate_list:
        c_first = pvalue_matrix.loc[c,:].dropna()
        c_second = pvalue_matrix.loc[:,c].dropna()
        U = n_candidates - np.sum(multipletests(c_first, alpha=alpha / 2, method=correction_method)[0])
        L = 1 + np.sum(multipletests(c_second, alpha=alpha / 2, method=correction_method)[0])
        all_ranks[c] = {'L': L, 'U': U}
    return pd.DataFrame(all_ranks).T


def ranking_from_multiple_paired_tests(
    paired_test_res: pd.DataFrame,
    alpha: float = 0.1,
    correction_method: str = 'holm',
    marginal_correction: bool = True,
    c1_col: str = 'candidate_1',
    c2_col: str = 'candidate_2',
) -> pd.DataFrame:
    """Convert pairwise test p-values to per-model rank intervals.

    Parameters
    ----------
    paired_test_res : DataFrame
        Output of ``calc_paired_tests``.
    alpha : float
        Significance level.
    correction_method : str
        Multiple-comparison method (default Holm).
    marginal_correction : bool
        If True (default), use ``calc_marginal_intervals``; otherwise
        ``calc_simultaneous_intervals``.
    c1_col, c2_col : str
        Directed pair column names.

    Returns
    -------
    DataFrame
        Columns ``L``, ``U`` indexed by model name.
    """
    set_ranks = None
    if marginal_correction:
        set_ranks = calc_marginal_intervals(paired_test_res, alpha=alpha, 
                                       c1_col=c1_col, c2_col=c2_col, 
                                       correction_method=correction_method)
    else:
        set_ranks = calc_simultaneous_intervals(paired_test_res, alpha=alpha, 
                                       c1_col=c1_col, c2_col=c2_col, 
                                       correction_method=correction_method)
    return set_ranks



# --- Bootstrap ranking intervals ---

def calc_bootstrap_scores(
    base_values: pd.DataFrame | np.ndarray,
    n_bootstrap: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Draw bootstrap replicate means for each model.

    Parameters
    ----------
    base_values : DataFrame or ndarray
        Paired scores, shape ``(n_observations, n_models)``.
    n_bootstrap : int
        Number of bootstrap replicates.
    seed : int, optional
        Passed to ``numpy.random.default_rng``.

    Returns
    -------
    DataFrame
        Shape ``(n_bootstrap, n_models)``; each row is a bootstrap mean vector.
    """
    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be at least 1.')

    data = _validate_base_values(data=base_values)
    columns = data.columns
    X = data.values.astype(np.float64)
    n_obs = X.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_obs, size=(n_bootstrap, n_obs))
    sampled = X[idx]
    bootstrap_scores = np.mean(sampled, axis=1)
    return pd.DataFrame(bootstrap_scores, columns=columns)


def _rank_rowwise_after_column_shuffle(bootstrap_scores, seed=None, ascending=True):
    """Assign ranks within each bootstrap replicate with random tie-breaking.

    Columns are shuffled independently per row before ranking so tied values
    receive distinct ranks at random (``method='first'``).

    Parameters
    ----------
    bootstrap_scores : DataFrame
        Shape ``(n_bootstrap, n_models)``.
    seed : int, optional
        Seeds a deterministic per-row RNG stream.
    ascending : bool
        If True, lowest score gets rank 1; if False, highest score gets rank 1.

    Returns
    -------
    DataFrame
        Integer ranks, same shape as ``bootstrap_scores``.
    """
    n_rows, n_cols = bootstrap_scores.shape
    row_sequences = np.random.SeedSequence(entropy=[seed, n_rows, n_cols]).spawn(n_rows)
    out = pd.DataFrame(index=bootstrap_scores.index, columns=bootstrap_scores.columns, dtype=int)

    for i in range(n_rows):
        rng = np.random.default_rng(row_sequences[i])
        row = bootstrap_scores.iloc[i]
        shuffled = row.sample(frac=1, random_state=rng, replace=False)
        ranked = shuffled.rank(method='first', ascending=ascending)
        out.iloc[i] = ranked.reindex(bootstrap_scores.columns)

    return out


def ranking_from_bootstrap_scores(
    bootstrap_scores: pd.DataFrame,
    alpha: float = 0.1,
    ascending: bool = True,
    seed: int | None = None,
) -> pd.DataFrame:
    """Rank intervals from bootstrap replicate ranks.

    Parameters
    ----------
    bootstrap_scores : DataFrame
        Bootstrap mean scores (e.g. from ``calc_bootstrap_scores``).
    alpha : float
        Two-sided interval level; uses ``alpha / 2`` in each tail.
    ascending : bool
        Ranking direction passed to ``rank_rowwise_after_column_shuffle``.
    seed : int, optional
        Tie-breaking seed for row-wise ranking.

    Returns
    -------
    DataFrame
        Columns ``L``, ``U`` per model (quantiles of bootstrap ranks).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError('alpha must lie in (0, 1).')

    ranks = _rank_rowwise_after_column_shuffle(bootstrap_scores, ascending=ascending, seed=seed)       
    q_lo = ranks.quantile(alpha / 2.0, axis=0,interpolation='lower').astype(int) # floor
    q_lo.name = 'L'
    q_hi = ranks.quantile(1 - (alpha / 2), axis=0, interpolation='higher').astype(int) # ceil
    q_hi.name = 'U'
    return pd.concat([q_lo, q_hi], axis=1)