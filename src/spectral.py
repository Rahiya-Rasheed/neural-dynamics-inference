"""Spectra, dominant rhythm, decimation factor, envelope shape."""
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.stats import kurtosis

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 25)}


def psd_all(trials, fs, nperseg=4096):
    """Channel-averaged Welch PSD per trial. Returns f, P of shape (n_trials, F)."""
    Ps = []
    for x in trials:
        f, P = welch(x, fs=fs, nperseg=min(nperseg, x.shape[1]), axis=1)
        Ps.append(P.mean(axis=0))
    return f, np.array(Ps)


def fit_peak(f, p, lo=4.0, hi=8.0, bgwin=41):
    """Peak center, FWHM, prominence — measured above a 1/f background
    (running median of log-PSD). Center refined by a 3-point parabola."""
    L = np.log10(p)
    R = L - median_filter(L, size=bgwin)
    idx = np.where((f >= lo) & (f <= hi))[0]
    i = idx[np.argmax(R[idx])]
    y = R[i-1:i+2]
    d = y[0] - 2*y[1] + y[2]
    fpk = f[i] - 0.5*(f[i+1]-f[i])*(y[2]-y[0])/d if d != 0 else f[i]
    half = R[i] / 2.0
    j = i
    while j > idx[0]  and R[j] > half: j -= 1
    k = i
    while k < idx[-1] and R[k] > half: k += 1
    return fpk, f[k]-f[j], R[i]


def choose_M(f, Pm, fs, frac=1e-3, Mmax=8):
    """Largest M leaving < frac of total power above the new Nyquist."""
    tot = np.trapezoid(Pm, f)
    for M in range(Mmax, 0, -1):
        sel = f >= fs/(2*M)
        if sel.any() and np.trapezoid(Pm[sel], f[sel])/tot < frac:
            return M
    return 1


def band_power_rel(f, P, lo=25.0, hi=60.0):
    """Per-trial [lo, hi] power relative to the cohort median."""
    sel = (f >= lo) & (f <= hi)
    bp = np.array([np.trapezoid(p[sel], f[sel]) for p in P])
    return bp / np.median(bp)


def envelope(x, fs, fpk, bw=1.5):
    b, a = butter(4, [(fpk-bw)/(fs/2), (fpk+bw)/(fs/2)], btype="band")
    return np.abs(hilbert(filtfilt(b, a, x)))


def bimodality_coefficient(e):
    """(skew² + 1)/(excess kurtosis + 3). Rayleigh (linear Gaussian) ≈ 0.43."""
    en = (e - e.mean()) / e.std()
    return (float((en**3).mean())**2 + 1) / (kurtosis(en) + 3)


def band_power_series(x, fs, win, lo, hi, nperseg=1024):
    """Band power of the channel average in consecutive windows."""
    K = x.shape[1] // win
    bp = []
    for k in range(K):
        fw, Pw = welch(x[:, k*win:(k+1)*win].mean(axis=0), fs=fs, nperseg=min(nperseg, win))
        s = (fw >= lo) & (fw < hi)
        bp.append(np.trapezoid(Pw[s], fw[s]))
    return np.array(bp)
