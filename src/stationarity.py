"""One time-invariant state, or several regimes?"""
import numpy as np
from scipy.stats import linregress
from .spectral import band_power_series

frob = lambda a, b: np.linalg.norm(a - b)


def win_covs(z, win):
    K = z.shape[1] // win
    return np.array([np.cov(z[:, k*win:(k+1)*win]) for k in range(K)])


def stationarity_sweep(Z, fs, windows_s=(0.5, 1, 2, 5, 10, 30)):
    """Rows [win_s, within median, between median, ratio]. Ratio converging
    with no plateau ⇒ windows are noisy estimates of one covariance, not regimes."""
    trial_covs = np.array([np.cov(z) for z in Z])
    n = len(Z)
    between = [frob(trial_covs[i], trial_covs[j]) for i in range(n) for j in range(i+1, n)]
    rows = []
    for sec in windows_s:
        win = int(sec * fs)
        within = []
        for z in Z:
            cs = win_covs(z, win)
            within += [frob(cs[i], cs[j]) for i in range(len(cs)) for j in range(i+1, len(cs))]
        if not within: continue                        # window longer than a trial
        rows.append([sec, np.median(within), np.median(between),
                     np.median(within) / np.median(between)])
    return np.array(rows)


def drift_tstats(trials, fs, bands, win_s=10.0):
    """(trial, band, t) for the slope of each band-power series vs window index."""
    win = int(win_s * fs)
    out = []
    for t, x in enumerate(trials):
        for name, (lo, hi) in bands.items():
            bp = band_power_series(x, fs, win, lo, hi)
            if len(bp) < 3: continue
            r = linregress(np.arange(len(bp)), bp)
            out.append((t, name, r.slope / r.stderr))
    return out
