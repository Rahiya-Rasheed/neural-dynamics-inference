"""Per-trial summary stats, outlier flags, peak events."""
import numpy as np
from scipy.stats import kurtosis

COLS = ["max|ch_mean|", "std_min", "std_med", "std_max",
        "cavg_resid", "min", "max", "kurt_med", "frac>5std"]


def hygiene_table(trials):
    """One row per trial. Confirms data arrived demeaned, referenced, artifact-cleaned."""
    rows = []
    for x in trials:
        ch_mean = x.mean(axis=1)
        ch_std  = x.std(axis=1)
        rows.append([
            np.abs(ch_mean).max(),
            ch_std.min(), np.median(ch_std), ch_std.max(),
            x.mean(axis=0).std(),                  # common-average residual
            x.min(), x.max(),
            np.median(kurtosis(x, axis=1)),
            (np.abs(x) > 5.0 * ch_std[:, None]).mean(),
        ])
    return np.array(rows)


def flag_outliers(trials, table):
    """Flag, never delete: exclusion is a decision to record, not automate."""
    flags = []
    med = np.median(table, axis=0)
    floor = 1e-9 * med[COLS.index("std_med")]   # below this a column is rounding error
    for i, row in enumerate(table):
        for j, col in enumerate(COLS):
            if col == "frac>5std":                      # count statistic:
                if row[j] > 10 * med[j]:                # only flag extreme excess
                    flags.append(f"trial {i}: {col} = {row[j]:.3g} vs median {med[j]:.3g}")
            elif med[j] > floor and not (0.5 <= row[j] / med[j] <= 2.0):
                flags.append(f"trial {i}: {col} = {row[j]:.3g} vs median {med[j]:.3g}")
    for i, x in enumerate(trials):
        s = x.std(axis=1)
        for ch in np.where((s < np.median(s)/2) | (s > np.median(s)*2))[0]:
            flags.append(f"trial {i}, channel {ch}: std = {s[ch]:.3g} "
                         f"(median {np.median(s):.3g})")
    return flags


def peak_event(x, fs, win_s=2.0, k=5.0):
    """Largest |sample| and how many channels exceed k·std within ±win_s.
    Many channels ⇒ montage-wide event, one channel ⇒ glitch."""
    ch, ts = np.unravel_index(np.abs(x).argmax(), x.shape)
    w = int(win_s * fs)
    seg = x[:, max(0, ts - w): ts + w]
    n_exceed = int((np.abs(seg).max(axis=1) > k * x.std(axis=1)).sum())
    return int(ch), ts/fs, float(x[ch, ts]), n_exceed
