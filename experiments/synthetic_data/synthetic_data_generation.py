"""
Synthetic data generation functions for simulation studies.
"""

import numpy as np
import pandas as pd


def generate_mean_vector(M, structure='increasing', **kwargs):
    """Generate a mean performance vector for M models with different structures.

    Parameters
    ----------
    M : int
        Number of models.
    structure : str
        Type of mean structure:

        - ``'increasing'``: increasing means in a sqrt-based pattern
          (default: ``[1, sqrt(2), ..., sqrt(M)]``).
        - ``'uniform'``: each mean drawn from ``Uniform(low, high)``
          (default: ``low=1``, ``high=sqrt(M)``).
        - ``'gamma'``: each mean drawn from ``Gamma(shape, scale)``
          (default: ``shape=1``, ``scale=1``).
    **kwargs
        Additional parameters:

        - For ``'increasing'``: ``low`` (default: 1), ``high`` (default: M).
        - For ``'uniform'``: ``low`` (default: 1), ``high`` (default: ``sqrt(M)``),
          ``seed`` (optional).
        - For ``'gamma'``: ``shape`` (default: 1), ``scale`` (default: 1),
          ``seed`` (optional).

    Returns
    -------
    mu : ndarray
        M-dimensional vector of mean performance values, ordered from worst
        (lowest mean) to best (highest mean).
    """
    if structure == 'increasing':
        # Increasing means in a sqrt-based pattern. Default: [1, sqrt(2), ..., sqrt(M)].
        low = kwargs.get('low', 1)
        high = kwargs.get('high', M)
        mu_base = np.linspace(low, high, M)
        mu = np.sqrt(mu_base)

    elif structure == 'uniform':
        # Each mean drawn from Uniform(low, high). Default: [1, sqrt(M)]
        low = kwargs.get('low', 1.0)
        high = kwargs.get('high', np.sqrt(M))
        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        mu = np.random.uniform(low=low, high=high, size=M)
    
    elif structure == 'gamma':
        # Each mean drawn from Gamma(shape, scale). Default: shape=1, scale=1.
        shape = kwargs.get('shape', 1.0)
        scale = kwargs.get('scale', 1.0)
        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        mu = np.random.gamma(shape=shape, scale=scale, size=M)
    
    else:
        raise ValueError(
            f'Unknown structure: {structure}. '
            "Choose from 'increasing', 'uniform', or 'gamma'."
        )
    
    return np.sort(mu)


def generate_correlation_matrix(M, rho=0.0, block_size=None):
    """Generate a correlation matrix R, either identity or block-wise.

    Parameters
    ----------
    M : int
        Number of models.
    rho : float, default 0.0
        Within-block correlation. When 0, returns the identity matrix.
    block_size : int or None
        Block size for block-wise correlation. Required when ``rho != 0``.

    Returns
    -------
    R : ndarray
        ``M x M`` correlation matrix.
    """
    R = np.eye(M) # No correlations: identity matrix

    if rho == 0.0:
        return R
    
    if block_size is None: # Block-wise correlation structure with rho != 0.0
        raise ValueError('block_size must be specified when rho != 0.0')
    
    start_idx = 0
    while start_idx < M:
        end_idx = min(start_idx + block_size, M)
        idx = np.arange(start_idx, end_idx)
        R[np.ix_(idx, idx)] = rho
        R[idx, idx] = 1.0
        start_idx = end_idx
    return R


def generate_sigma(M, sigma_mean=1.0, homoscedastic=True, seed=None):
    """Generate standard deviation vector for M models.

    Parameters
    ----------
    M : int
        Number of models.
    sigma_mean : float, default 1.0
        Mean value of sigma (fixed when homoscedastic; center of range when
        heteroscedastic).
    homoscedastic : bool, default True
        If True, use fixed sigma for all models. If False, sample sigma per
        model from a log-uniform distribution in
        ``[sigma_mean/2, sigma_mean*2]``.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    sigma : ndarray
        Standard deviation vector (M-dimensional).
    """
    if seed is not None:
        np.random.seed(seed)
    
    if homoscedastic:
        sigma = np.full(M, sigma_mean)
    else:
        # Heteroscedastic: log-uniform in [sigma_mean/2, sigma_mean*2], based on sigma_mean only
        log_low = np.log(sigma_mean / 2)
        log_high = np.log(sigma_mean * 2)
        sigma = np.exp(np.random.uniform(log_low, log_high, M))
    
    return sigma


def build_covariance_matrix(R, sigma):
    """Build covariance matrix from correlation matrix and standard deviations.

    Parameters
    ----------
    R : ndarray
        Correlation matrix (``M x M``).
    sigma : float or ndarray
        Standard deviation(s). If scalar, same for all models. If array,
        per-model.

    Returns
    -------
    Sigma : ndarray
        Covariance matrix (``M x M``). If ``sigma`` is scalar,
        ``Sigma = sigma**2 * R``. If ``sigma`` is an array,
        ``Sigma = D @ R @ D`` where ``D = diag(sigma)``.
    """
    if isinstance(sigma, np.ndarray):
        # Heteroscedastic case: $\\Sigma = D R D$ where $D = \\text{diag}(\\sigma_1, ..., \\sigma_M)$
        D = np.diag(sigma)
        Sigma = D @ R @ D
    else:
        # Homoscedastic case: $\\Sigma = \\sigma^2 R$
        Sigma = (sigma ** 2) * R
    
    return Sigma


def _introduce_ties(theta, tie_proportion):
    """Create ties per column by merging the closest values to their mean.

    Parameters
    ----------
    theta : ndarray
        Performance matrix (``M x N``): rows = models, columns = tasks.
    tie_proportion : float
        Value in ``[0, 1]``. ``0`` = no ties. ``1`` = tie all M values in the
        column (set to column mean). In between: tie the
        ``k = round(tie_proportion * M)`` closest values in the column
        (replace them by their mean). The closest k are the contiguous block
        of size k in sorted order with smallest range.

    Returns
    -------
    ndarray
        Copy of ``theta`` with ties introduced (within each column).
    """
    theta = np.asarray(theta, dtype=float).copy()
    if tie_proportion <= 0:
        return theta
    if tie_proportion >= 1:
        theta[:] = np.mean(theta, axis=0)
        return theta
    M, N = theta.shape
    k = min(int(round(tie_proportion * M)), M)
    k = max(2, k)  # at least 2 to form a tie
    for b in range(N):
        col = theta[:, b]
        order = np.argsort(col)
        s = col[order]  # sorted values
        ranges = s[k - 1 :] - s[: M - k + 1]
        best_start = np.argmin(ranges)
        tie_idx = order[best_start : best_start + k]
        theta[tie_idx, b] = np.mean(col[tie_idx])
    return theta


def generate_true_performance_matrix(mu, Sigma, N, tie_proportion=0.0, seed=None):
    """Generate true performance matrix from a multivariate normal distribution.

    Each column ``Theta[:, b] ~ N_M(mu, Sigma)`` independently via a Cholesky
    factor. Optionally introduces ties per column using ``_introduce_ties``.

    Parameters
    ----------
    mu : ndarray
        Mean vector (M-dimensional).
    Sigma : ndarray
        Covariance matrix (``M x M``).
    N : int
        Number of tasks (columns).
    tie_proportion : float, default 0.0
        Proportion of elements per column to tie (0 = none, 1 = all).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Theta : ndarray
        True performance matrix (``M x N``). Each column is independently
        sampled from ``N_M(mu, Sigma)``.
    """
    if seed is not None:
        np.random.seed(seed)

    M = mu.shape[0]

    # Ensure positive definiteness (add small regularization if needed)
    eigenvals = np.linalg.eigvals(Sigma)
    min_eigenval = np.min(eigenvals)
    if min_eigenval <= 0:
        regularization = abs(min_eigenval) + 1e-8
        Sigma = Sigma + regularization * np.eye(M)

    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        eigenvals, eigenvecs = np.linalg.eigh(Sigma)
        eigenvals = np.maximum(eigenvals, 1e-10)
        L = eigenvecs @ np.diag(np.sqrt(eigenvals))

    Z = np.random.randn(M, N)
    Theta = mu[:, np.newaxis] + L @ Z

    if tie_proportion > 0.0:
        Theta = _introduce_ties(Theta, tie_proportion)

    return Theta


def build_covariance_matrix_task(R, sigma, task_effect=0.1):
    """Build task-level covariance matrix from correlation and standard deviations.

    Parameters
    ----------
    R : ndarray
        Correlation matrix (``M x M``).
    sigma : float or ndarray
        Standard deviation(s). If scalar, same for all models. If array,
        per-model.
    task_effect : float, default 0.1
        Correlation substituted for off-block zeros in ``R``.

    Returns
    -------
    Sigma_task : ndarray
        Covariance matrix for task-level data (``M x M``).
    """
    R_task = np.where(R == 0, task_effect, R)
    return build_covariance_matrix(R_task, sigma)


def generate_observed_performance_data(Theta, Sigma_task, n_base, seed=None):
    """Generate observed performance data from a true performance matrix.

    Creates a DataFrame in long format with columns ``observation_id``,
    ``model``, ``task``, and ``score``.

    Parameters
    ----------
    Theta : ndarray
        True performance matrix (``M x N``).
    Sigma_task : ndarray
        Task-level covariance matrix (``M x M``) for observation noise.
    n_base : int
        Number of observations per model-task pair.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        Long format with columns ``observation_id``, ``model``, ``task``,
        ``score``.
    """
    if seed is not None:
        np.random.seed(seed)
    
    M, N = Theta.shape

    # Vectorized: X_{i,j,b} = Θ_{j,b} + ε_{i,j,b}, ε ~ N(0, Σ_task)
    noise = np.random.multivariate_normal(np.zeros(M), Sigma_task, (n_base, N)).reshape(n_base, M, N)
    X = (Theta[np.newaxis, :, :] + noise).ravel()
    
    # Index columns in same order as original (i, j, b) nested loops
    observation_id = np.repeat(np.arange(n_base), M * N)
    model_idx = np.tile(np.repeat(np.arange(M), N), n_base)
    task_idx = np.tile(np.arange(N), n_base * M)
    
    # Precompute labels (M + N strings) and index instead of building n_rows strings
    model_labels = np.array([f'model_{j}' for j in range(M)], dtype=object)
    task_labels = np.array([f'task_{b}' for b in range(N)], dtype=object)
    
    return pd.DataFrame({
        'observation_id': observation_id,
        'model': model_labels[model_idx],
        'task': task_labels[task_idx],
        'score': X,
    }, columns=['observation_id', 'model', 'task', 'score'])
