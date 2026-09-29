"""Load, save and decimate EEG trials."""
from dataclasses import dataclass
import numpy as np
import scipy.io as sio
from scipy.signal import decimate


@dataclass
class Config:
    mat_path: str
    fs_assumed: float = 250.0   # not confirmed for the recordings
    n_trials: int = 16
    n_channels: int = 61
    decimate_by: int = 1

    @property
    def fs(self) -> float:
        return self.fs_assumed / self.decimate_by


def load_trials(cfg):
    """Return n_trials arrays (n_channels, T), float64, channels-first. Fails loudly."""
    m = sio.loadmat(cfg.mat_path)
    if "Xcell" not in m:
        raise KeyError(f"expected key 'Xcell', found {list(m)}")
    trials = [np.asarray(x, dtype=np.float64) for x in m["Xcell"][0]]
    if len(trials) != cfg.n_trials:
        raise ValueError(f"expected {cfg.n_trials} trials, got {len(trials)}")
    for i, x in enumerate(trials):
        if x.ndim != 2 or x.shape[0] != cfg.n_channels:
            raise ValueError(f"trial {i}: shape {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"trial {i}: non-finite values")
    return trials


def save_trials(path, trials):
    """Same 'Xcell' layout load_trials reads."""
    cell = np.empty((1, len(trials)), dtype=object)
    for i, x in enumerate(trials): cell[0, i] = x
    sio.savemat(path, {"Xcell": cell})


def decimate_trial(x, M):
    """Anti-aliased downsampling by M. Never x[:, ::M] — that folds
    broadband noise above the new Nyquist back into band."""
    return x if M == 1 else decimate(x, M, axis=1, ftype="fir", zero_phase=True)
