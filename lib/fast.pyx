import numpy as np
cimport numpy as np
from libc.math cimport fabs, NAN, isfinite, sqrt
from scipy.stats import rankdata


def rank(alpha):
    v = np.isfinite(alpha)
    alpha[v] = (rankdata(alpha[v]) - 1.) / (v.sum() - 1.)
    return alpha


def power(alpha, exponent=1):
    v = np.isfinite(alpha)
    if v.sum() == 1:
        alpha[v] = 0.
    else:
        alpha[v] = (rankdata(alpha[v]) - 1.) / (v.sum() - 1.) - 0.5
        if exponent != 1:
            alpha[v] = np.sign(alpha[v]) * np.abs(alpha[v]) ** exponent
    return alpha


def nioscale(alpha, booksize):
    alpha = alpha.copy()
    if booksize == 0:
        return alpha
    cbook = np.nansum(np.abs(alpha))
    if cbook > 0.:
        alpha *= (booksize / cbook)
    return alpha


def winsorize(alpha, sigma=3.0):
    v = np.isfinite(alpha)
    if v.sum() < 2:
        return alpha

    alpha_valid = alpha[v]
    mean = np.nanmean(alpha_valid)
    std = np.nanstd(alpha_valid)
    
    if std <= 0.0:
        return alpha
    
    lower_bound = mean - sigma * std
    upper_bound = mean + sigma * std
    
    alpha[v] = np.clip(alpha_valid, lower_bound, upper_bound)
    return alpha


def zscore(alpha):
    v = np.isfinite(alpha)
    if v.sum() < 2:
        return alpha
    
    alpha_valid = alpha[v]
    mean = np.nanmean(alpha_valid)
    std = np.nanstd(alpha_valid)
    
    if std <= 0.0:
        alpha[v] = np.nan
    else:
        alpha[v] = (alpha_valid - mean) / std
    
    return alpha


def niotsrank(const np.float32_t[:, :] data):
    cdef Py_ssize_t n_itvls = data.shape[0], n_symbols = data.shape[1]
    cdef np.float32_t larger, smaller, a, a0
    total = np.zeros(n_symbols, dtype=np.float32)
    for j in range(n_symbols):
        a0 = data[0, j]
        if isfinite(a0):
            total[j] = 1.
            larger, smaller = 0, 0
            for i in range(1, n_itvls):
                a = data[i, j]
                if isfinite(a):
                    total[j] += 1
                    if (a > a0) and (fabs(a - a0) > (fabs(a + a0) * 1e-9)):
                        larger += 1
                    elif (a < a0) and (fabs(a - a0) > (fabs(a + a0) * 1e-9)):
                        smaller += 1
            if total[j] == 1:
                total[j] = NAN
            else:
                a0 = (total[j] - larger - smaller - 1.) / 2.
                total[j] = (smaller + a0) / (total[j] - 1.)
        else:
            total[j] = NAN
    return total


#def sum(const np.float32_t[:, :] data):
#    cdef Py_ssize_t n_itvls = data.shape[0], n_symbols = data.shape[1], n
#    cdef np.float32_t a
#    res = np.zeros(n_symbols, dtype=np.float32)
#    for j in range(n_symbols):
#        n = 0
#        for i in range(n_itvls):
#            a = data[i, j]
#            if isfinite(a):
#                res[j] += a
#                n += 1
#        if n == 0:
#            res[j] = NAN
#    return res
def niosum(const np.float32_t[:, :] data):
    v = np.any(np.isfinite(data), axis=0)
    return np.where(v, np.nansum(data, axis=0), np.nan)


#def mean(const np.float32_t[:, :] data):
#    cdef Py_ssize_t n_itvls = data.shape[0], n_symbols = data.shape[1]
#    cdef np.float32_t a
#    res = np.zeros(n_symbols, dtype=np.float32)
#    counts = np.zeros(n_symbols, dtype=np.int32)
#    for j in range(n_symbols):
#        for i in range(n_itvls):
#            a = data[i, j]
#            if isfinite(a):
#                res[j] += a
#                counts[j] += 1
#        if counts[j] == 0:
#            res[j] = NAN
#        else:
#            res[j] /= counts[j]
#    return res
def niomean(const np.float32_t[:, :] data):
    return np.nanmean(data, axis=0)


#def std(const np.float32_t[:, :] data):
#    cdef Py_ssize_t n_itvls = data.shape[0], n_symbols = data.shape[1]
#    cdef np.float32_t m, n, a
#    res = np.zeros(n_symbols, dtype=np.float32)
#    for j in range(n_symbols):
#        m, n = 0., 0.
#        for i in range(n_itvls):
#            a = data[i, j]
#            if isfinite(a):
#                m += a
#                n += 1
#        if n < 2:
#            res[j] = NAN
#        else:
#            m /= n
#            for i in range(n_itvls):
#                a = data[i, j]
#                if isfinite(a):
#                    a -= m
#                    res[j] += a * a
#            res[j] /= (n - 1)
#            res[j] = sqrt(res[j])
#    return res
def niostd(const np.float32_t[:, :] data):
    return np.nanstd(data, axis=0)


def niostdup(const np.float32_t[:, :] data):
    """Calculate upside volatility (standard deviation of values above mean)."""
    cdef Py_ssize_t n_itvls = data.shape[0], n_symbols = data.shape[1]
    cdef np.float32_t m, n, a, n_up, x2
    res = np.zeros(n_symbols, dtype=np.float32)
    for j in range(n_symbols):
        m, n = 0., 0.
        # First pass: calculate mean
        for i in range(n_itvls):
            a = data[i, j]
            if isfinite(a):
                m += a
                n += 1
        if n < 2:
            res[j] = NAN
        else:
            m /= n
            # Second pass: calculate std of values above mean
            x2, n_up = 0., 0.
            for i in range(n_itvls):
                a = data[i, j]
                if isfinite(a) and a > m:
                    a -= m
                    x2 += a * a
                    n_up += 1
            if n_up < 2:
                res[j] = NAN
            else:
                res[j] = sqrt(x2 / (n_up - 1))
    return res


def niostddown(const np.float32_t[:, :] data):
    """Calculate downside volatility (standard deviation of values below mean)."""
    cdef Py_ssize_t n_itvls = data.shape[0], n_symbols = data.shape[1]
    cdef np.float32_t m, n, a, n_down, x2
    res = np.zeros(n_symbols, dtype=np.float32)
    for j in range(n_symbols):
        m, n = 0., 0.
        # First pass: calculate mean
        for i in range(n_itvls):
            a = data[i, j]
            if isfinite(a):
                m += a
                n += 1
        if n < 2:
            res[j] = NAN
        else:
            m /= n
            # Second pass: calculate std of values below mean
            x2, n_down = 0., 0.
            for i in range(n_itvls):
                a = data[i, j]
                if isfinite(a) and a < m:
                    a -= m
                    x2 += a * a
                    n_down += 1
            if n_down < 2:
                res[j] = NAN
            else:
                res[j] = sqrt(x2 / (n_down - 1))
    return res


def nioskew(const np.float32_t[:, :] data):
    cdef Py_ssize_t n_itvls = data.shape[0], n_symbols = data.shape[1]
    cdef np.float32_t m, n, a, a2, a3, x2, x3
    res = np.zeros(n_symbols, dtype=np.float32)
    for j in range(n_symbols):
        m, n = 0., 0.
        for i in range(n_itvls):
            a = data[i, j]
            if isfinite(a):
                m += a
                n += 1
        if n < 2:
            res[j] = NAN
        else:
            m /= n
            x2, x3 = 0., 0.
            for i in range(n_itvls):
                a = data[i, j]
                if isfinite(a):
                    a -= m
                    a2 = a * a
                    a3 = a2 * a
                    x2 += a2
                    x3 += a3
            a = x2 / (n - 1)
            a = a * sqrt(a)
            res[j] = x3 / n / a if a > 0 else NAN
    return res


def niokurt(const np.float32_t[:, :] data):
    cdef Py_ssize_t n_itvls = data.shape[0], n_symbols = data.shape[1]
    cdef np.float32_t m, n, a, a2, a4, x2, x4
    res = np.zeros(n_symbols, dtype=np.float32)
    for j in range(n_symbols):
        m, n = 0., 0.
        for i in range(n_itvls):
            a = data[i, j]
            if isfinite(a):
                m += a
                n += 1
        if n < 2:
            res[j] = NAN
        else:
            m /= n
            x2, x4 = 0., 0.
            for i in range(n_itvls):
                a = data[i, j]
                if isfinite(a):
                    a -= m
                    a2 = a * a
                    a4 = a2 * a2
                    x2 += a2
                    x4 += a4
            a = x2 / (n - 1)
            a = a * a
            res[j] = x4 / n / a if a > 0 else NAN
    return res


def niobeta(const np.float32_t[:, :] datax, const np.float32_t[:, :] datay):
    cdef Py_ssize_t n_itvls = datax.shape[0], n_symbols = datax.shape[1]
    cdef np.float32_t mx, my, sx, sy, n, x, y
    res = np.zeros(n_symbols, dtype=np.float32)
    for j in range(n_symbols):
        mx, my, n = 0., 0., 0.
        for i in range(n_itvls):
            x, y = datax[i, j], datay[i, j]
            if isfinite(x) and isfinite(y):
                mx += x
                my += y
                n += 1
        if n < 2:
            res[j] = NAN
            continue
        mx /= n
        my /= n
        sx, sy = 0., 0.
        for i in range(n_itvls):
            x, y = datax[i, j], datay[i, j]
            if isfinite(x) and isfinite(y):
                x -= mx
                y -= my
                sx += x * x
                sy += x * y
        res[j] = sy / sx if sx > 0 else NAN
    return res


def niocov(const np.float32_t[:, :] datax, const np.float32_t[:, :] datay):
    cdef Py_ssize_t n_itvls = datax.shape[0], n_symbols = datax.shape[1]
    cdef np.float32_t mx, my, s, n, x, y
    res = np.zeros(n_symbols, dtype=np.float32)
    for j in range(n_symbols):
        mx, my, n = 0., 0., 0.
        for i in range(n_itvls):
            x, y = datax[i, j], datay[i, j]
            if isfinite(x) and isfinite(y):
                mx += x
                my += y
                n += 1
        if n < 2:
            res[j] = NAN
            continue
        mx /= n
        my /= n
        s = 0.
        for i in range(n_itvls):
            x, y = datax[i, j], datay[i, j]
            if isfinite(x) and isfinite(y):
                x -= mx
                y -= my
                s += x * y
        res[j] = s / (n - 1)
    return res


def niocorr(const np.float32_t[:, :] datax, const np.float32_t[:, :] datay):
    cdef Py_ssize_t n_itvls = datax.shape[0], n_symbols = datax.shape[1]
    cdef np.float32_t mx, my, sx, sy, sxy, n, x, y
    res = np.zeros(n_symbols, dtype=np.float32)
    for j in range(n_symbols):
        mx, my, n = 0., 0., 0.
        for i in range(n_itvls):
            x, y = datax[i, j], datay[i, j]
            if isfinite(x) and isfinite(y):
                mx += x
                my += y
                n += 1
        if n < 2:
            res[j] = NAN
            continue
        mx /= n
        my /= n
        sx, sy, sxy = 0., 0., 0.
        for i in range(n_itvls):
            x, y = datax[i, j], datay[i, j]
            if isfinite(x) and isfinite(y):
                x -= mx
                y -= my
                sx += x * x
                sy += y * y
                sxy += x * y
        sx = sx * sy
        res[j] = sxy / sqrt(sx) if sx > 0 else NAN
    return res


def niospcorr(const np.float32_t[:, :] datax, const np.float32_t[:, :] datay):
    """
    Calculate Spearman correlation coefficient between datax and datay.
    For each symbol, computes ranks and then Pearson correlation on ranks.
    Optimized implementation using efficient ranking algorithm.
    """
    cdef Py_ssize_t n_itvls = datax.shape[0], n_symbols = datax.shape[1]
    cdef Py_ssize_t i, j, k, n, start, end
    cdef np.float32_t mx, my, sx, sy, sxy, x, y, rank_x, rank_y
    cdef np.float32_t tie_rank
    res = np.zeros(n_symbols, dtype=np.float32)
    
    # Temporary arrays
    cdef np.float32_t[:] valid_x = np.zeros(n_itvls, dtype=np.float32)
    cdef np.float32_t[:] valid_y = np.zeros(n_itvls, dtype=np.float32)
    cdef Py_ssize_t[:] idx_x = np.zeros(n_itvls, dtype=np.intp)
    cdef Py_ssize_t[:] idx_y = np.zeros(n_itvls, dtype=np.intp)
    cdef np.float32_t[:] ranks_x = np.zeros(n_itvls, dtype=np.float32)
    cdef np.float32_t[:] ranks_y = np.zeros(n_itvls, dtype=np.float32)
    cdef Py_ssize_t temp_idx
    
    for j in range(n_symbols):
        # Collect valid pairs
        n = 0
        for i in range(n_itvls):
            x, y = datax[i, j], datay[i, j]
            if isfinite(x) and isfinite(y):
                valid_x[n] = x
                valid_y[n] = y
                idx_x[n] = n
                idx_y[n] = n
                n += 1
        
        if n < 2:
            res[j] = NAN
            continue
        
        # Sort x values using insertion sort (simple and efficient for small arrays)
        for i in range(1, n):
            k = i
            while k > 0 and valid_x[idx_x[k-1]] > valid_x[idx_x[k]]:
                temp_idx = idx_x[k]
                idx_x[k] = idx_x[k-1]
                idx_x[k-1] = temp_idx
                k -= 1
        
        # Assign ranks for x (handle ties by averaging)
        i = 0
        while i < n:
            start = i
            # Find all ties
            while i < n - 1 and valid_x[idx_x[i]] == valid_x[idx_x[i+1]]:
                i += 1
            end = i
            # Average rank for tied values
            tie_rank = (start + end + 2) / 2.0  # +2 because ranks start at 1
            for k in range(start, end + 1):
                ranks_x[idx_x[k]] = tie_rank
            i += 1
        
        # Sort y values
        for i in range(1, n):
            k = i
            while k > 0 and valid_y[idx_y[k-1]] > valid_y[idx_y[k]]:
                temp_idx = idx_y[k]
                idx_y[k] = idx_y[k-1]
                idx_y[k-1] = temp_idx
                k -= 1
        
        # Assign ranks for y (handle ties by averaging)
        i = 0
        while i < n:
            start = i
            # Find all ties
            while i < n - 1 and valid_y[idx_y[i]] == valid_y[idx_y[i+1]]:
                i += 1
            end = i
            # Average rank for tied values
            tie_rank = (start + end + 2) / 2.0  # +2 because ranks start at 1
            for k in range(start, end + 1):
                ranks_y[idx_y[k]] = tie_rank
            i += 1
        
        # Calculate Pearson correlation on ranks
        mx, my = 0., 0.
        for i in range(n):
            mx += ranks_x[i]
            my += ranks_y[i]
        mx /= n
        my /= n
        
        sx, sy, sxy = 0., 0., 0.
        for i in range(n):
            x = ranks_x[i] - mx
            y = ranks_y[i] - my
            sx += x * x
            sy += y * y
            sxy += x * y
        
        sx = sx * sy
        res[j] = sxy / sqrt(sx) if sx > 0 else NAN
    
    return res


def nioregress(const np.float32_t[:, :] datax, const np.float32_t[:, :] datay):
    cdef Py_ssize_t n_itvls = datax.shape[0], n_symbols = datax.shape[1]
    cdef np.float32_t mx, my, sx, sy, n, x, y
    alpha = np.zeros(n_symbols, dtype=np.float32)
    beta = np.zeros(n_symbols, dtype=np.float32)
    for j in range(n_symbols):
        mx, my, n = 0., 0., 0.
        for i in range(n_itvls):
            x, y = datax[i, j], datay[i, j]
            if isfinite(x) and isfinite(y):
                mx += x
                my += y
                n += 1
        if n < 2:
            beta[j] = NAN
            alpha[j] = NAN
            continue
        mx /= n
        my /= n
        sx, sy = 0., 0.
        for i in range(n_itvls):
            x, y = datax[i, j], datay[i, j]
            if isfinite(x) and isfinite(y):
                x -= mx
                y -= my
                sx += x * x
                sy += x * y
        beta[j] = sy / sx if sx > 0 else NAN
        alpha[j] = my - beta[j] * mx
    return alpha, beta


def niolla(const np.float32_t[:, :] datax, const np.float32_t[:, :] datay,
           float halflife, float beta, bint mode):
    """Leader-lagger adjust (LLA) via flip history accumulation.
    datax: leader prices (n_itvls, n_symbols), chronological (0=oldest)
    datay: lagger prices (n_itvls, n_symbols)
    Returns: leader_fh, lagger_fh, fh_diff — each (n_symbols,)."""
    cdef Py_ssize_t n_itvls = datax.shape[0], n_symbols = datax.shape[1]
    cdef Py_ssize_t i, j
    cdef np.float32_t decay, lm, fm, lf, ff, min_fh, max_fh, x0, x1, y0, y1

    decay = <np.float32_t>(0.5 ** (1.0 / halflife))

    leader_fh = np.zeros(n_symbols, dtype=np.float32)
    lagger_fh = np.zeros(n_symbols, dtype=np.float32)
    fh_diff = np.zeros(n_symbols, dtype=np.float32)

    for i in range(1, n_itvls):
        for j in range(n_symbols):
            x1 = datax[i, j]
            x0 = datax[i - 1, j]
            y1 = datay[i, j]
            y0 = datay[i - 1, j]
            if not (isfinite(x1) and isfinite(x0) and isfinite(y1) and isfinite(y0)):
                continue
            if mode:
                lm = x1 - x0
                fm = y1 - y0
            else:
                if fabs(x0) < 1e-12 or fabs(y0) < 1e-12:
                    continue
                lm = (x1 - x0) / x0
                fm = (y1 - y0) / y0

            lf = leader_fh[j] * decay + lm * beta
            ff = lagger_fh[j] * decay + fm

            min_fh = 0.0
            max_fh = 0.0
            if lf > 0 and ff > 0:
                min_fh = lf if lf < ff else ff
            if lf < 0 and ff < 0:
                max_fh = lf if lf > ff else ff
            lf = lf - min_fh - max_fh
            ff = ff - min_fh - max_fh

            leader_fh[j] = lf
            lagger_fh[j] = ff
            fh_diff[j] = lf - ff

    return leader_fh, lagger_fh, fh_diff
