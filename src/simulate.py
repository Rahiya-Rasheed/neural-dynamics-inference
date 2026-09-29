"""Synthetic stand-in for the recordings, known parameters.

3-d latent SDE (damped rotation + slow OU), mixed into channels, sensor noise,
low-pass, common-average reference, demeaned. Same parameters every trial.
"""
import numpy as np
from scipy.signal import butter, filtfilt
from .continuous import psd_sqrt, van_loan


def latent_drift(f_rot=6.0, decay=4.0, slow_rate=1.0):
    w = 2*np.pi*f_rot
    return np.array([[-decay, -w, 0.0],
                     [w, -decay, 0.0],
                     [0.0, 0.0, -slow_rate]])


def synthetic_trials(n_trials=16, n_channels=61, duration_s=20.0, fs=250.0,
                     f_rot=6.0, decay=4.0, sensor_std=0.5, lowpass_hz=40.0, seed=0):
    rng = np.random.default_rng(seed)
    Ac = latent_drift(f_rot, decay)
    Qc = np.diag([4.0*decay, 4.0*decay, 2.0])          # -> stationary variance ~1-2
    Ad, Qd = van_loan(Ac, Qc, 1.0/fs); Lq = psd_sqrt(Qd)
    mixing = rng.normal(size=(n_channels, 3))
    mixing /= np.linalg.norm(mixing, axis=1, keepdims=True)   # similar channel scales
    b, a = butter(4, lowpass_hz/(fs/2))
    T = int(duration_s * fs)
    trials = []
    for _ in range(n_trials):
        z = np.zeros((3, T))
        for k in range(1, T): z[:, k] = Ad @ z[:, k-1] + Lq @ rng.standard_normal(3)
        x = filtfilt(b, a, mixing @ z + sensor_std*rng.standard_normal((n_channels, T)), axis=1)
        x -= x.mean(axis=0, keepdims=True)             # common-average reference
        x -= x.mean(axis=1, keepdims=True)             # demean
        trials.append(x)
    return trials
