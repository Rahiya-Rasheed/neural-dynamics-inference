"""Channel repair, PCA, held-out dimensionality, correlation time."""
import numpy as np
from scipy.signal import decimate


def interpolate_channels(trials, exclude):
    """exclude = {trial: [channels]}. Replace with the median of the other channels
    so every trial keeps all channels and one projection fits all trials."""
    out = [x.copy() for x in trials]
    for t, chans in exclude.items():
        keep = np.setdiff1d(np.arange(trials[t].shape[0]), chans)
        out[t][chans] = np.median(trials[t][keep], axis=0)
    return out


def pca_basis(trials):
    """Eigenvalues (descending) and eigenvectors of the pooled channel covariance."""
    w, V = np.linalg.eigh(np.cov(np.concatenate(trials, axis=1)))
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def variance_captured(train, test, d_max=10):
    """Rows [d, in-sample %, held-out %] for the top-d training PCs."""
    Ctr = np.cov(np.concatenate(train, axis=1))
    Cte = np.cov(np.concatenate(test,  axis=1))
    _, V = pca_basis(train)
    rows = []
    for d in range(1, d_max + 1):
        U = V[:, :d]
        rows.append([d,
                     100*np.trace(U.T @ Ctr @ U)/np.trace(Ctr),   # in-sample
                     100*np.trace(U.T @ Cte @ U)/np.trace(Cte)])  # held-out
    return np.array(rows)


def tau_c(z, fs, maxlag_s=5.0):
    """Integrated |autocorrelation|, seconds."""
    zc = (z - z.mean()) / z.std()
    L = int(maxlag_s * fs)
    ac = np.array([1.0] + [np.mean(zc[:-l] * zc[l:]) for l in range(1, L)])
    return np.abs(ac).sum() / fs


def project_decimate(trials, P, M):
    """Project onto rows of P, anti-aliased decimation by M, standardize each series."""
    Z = [P @ x for x in trials]
    if M > 1: Z = [decimate(z, M, axis=1, ftype="fir", zero_phase=True) for z in Z]
    return [(z - z.mean(1, keepdims=True)) / z.std(1, keepdims=True) for z in Z]
