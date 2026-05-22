"""Pure Python numerical utilities (no Cython dependency)."""
import numpy as np
from scipy.stats import rankdata


def rank(alpha):
    """Cross-sectional rank, output in [0, 1]."""
    v = np.isfinite(alpha)
    n = v.sum()
    if n <= 1:
        alpha[v] = 0.5
    else:
        alpha[v] = (rankdata(alpha[v]) - 1.0) / (n - 1.0)
    return alpha


def power(alpha, exponent=1.0):
    """Rank-based power transform, output centered at 0."""
    v = np.isfinite(alpha)
    n = v.sum()
    if n <= 1:
        alpha[v] = 0.0
    else:
        alpha[v] = (rankdata(alpha[v]) - 1.0) / (n - 1.0) - 0.5
        if exponent != 1.0:
            alpha[v] = np.sign(alpha[v]) * np.abs(alpha[v]) ** exponent
    return alpha


def nioscale(alpha, booksize):
    """Scale alpha to target book size (sum of abs values)."""
    alpha = alpha.copy()
    if booksize == 0:
        return alpha
    cbook = np.nansum(np.abs(alpha))
    if cbook > 0.0:
        alpha *= booksize / cbook
    return alpha


def niomean(data):
    """Column-wise nanmean, returns 1D array."""
    return np.nanmean(data, axis=0)


def niosum(data):
    """Column-wise nansum, returns 1D array."""
    return np.nansum(data, axis=0)


def winsorize(alpha, sigma=3.0):
    """Winsorize at +/- sigma standard deviations."""
    v = np.isfinite(alpha)
    if v.sum() < 2:
        return alpha
    mean = np.nanmean(alpha[v])
    std = np.nanstd(alpha[v])
    if std <= 0.0:
        return alpha
    lower = mean - sigma * std
    upper = mean + sigma * std
    alpha[v] = np.clip(alpha[v], lower, upper)
    return alpha


def zscore(alpha):
    """Cross-sectional z-score."""
    v = np.isfinite(alpha)
    if v.sum() < 2:
        return alpha
    mean = np.nanmean(alpha[v])
    std = np.nanstd(alpha[v])
    if std <= 0.0:
        alpha[v] = np.nan
    else:
        alpha[v] = (alpha[v] - mean) / std
    return alpha


def niobeta(datax, datay):
    """Compute beta of each column in datay vs datax.

    datax: (n_itvls, n_symbols) - independent variable (e.g., BTC returns)
    datay: (n_itvls, n_symbols) - dependent variable (e.g., each symbol's returns)
    Returns: (n_symbols,) beta values = cov(x,y) / var(x) per column.
    """
    n = datax.shape[0]
    # Mean
    valid = np.isfinite(datax) & np.isfinite(datay)
    count = valid.sum(axis=0).astype(np.float32)
    datax_masked = np.where(valid, datax, 0.0)
    datay_masked = np.where(valid, datay, 0.0)
    mx = datax_masked.sum(axis=0) / np.maximum(count, 1)
    my = datay_masked.sum(axis=0) / np.maximum(count, 1)
    # Cov / Var
    dx = np.where(valid, datax - mx, 0.0)
    dy = np.where(valid, datay - my, 0.0)
    var_x = (dx * dx).sum(axis=0)
    cov_xy = (dx * dy).sum(axis=0)
    beta = np.where(var_x > 0, cov_xy / var_x, np.nan)
    beta[count < 2] = np.nan
    return beta.astype(np.float32)
