"""Hygiene, spectra, stationarity, dimensionality. Everything lands in --out."""
import argparse, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hygiene import COLS, flag_outliers, hygiene_table, peak_event
from src.io import Config, load_trials
from src.pipeline import parse_exclude
from src.projection import interpolate_channels, pca_basis, tau_c, variance_captured
from src.spectral import (BANDS, band_power_rel, band_power_series, bimodality_coefficient,
                          choose_M, envelope, fit_peak, psd_all)
from src.stationarity import drift_tstats, stationarity_sweep

ap = argparse.ArgumentParser()
ap.add_argument("--mat", required=True)
ap.add_argument("--out", default="results/characterize")
ap.add_argument("--fs", type=float, default=250.0, help="assumed sampling rate")
ap.add_argument("--exclude", nargs="*", default=[], help="trial:channel[,channel], e.g. 8:46")
ap.add_argument("--n-train", type=int, default=8)
ap.add_argument("--d", type=int, default=3)
ap.add_argument("--env-channel", type=int, default=0)
args = ap.parse_args()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
cfg = Config(mat_path=args.mat, fs_assumed=args.fs)
trials = load_trials(cfg); fs = cfg.fs; n = len(trials)
print(n, "trials,", sorted({x.shape for x in trials}))

# --- hygiene: flags only, exclusions come from --exclude ---
table = hygiene_table(trials)
pd.DataFrame(table, columns=COLS).rename_axis("trial").to_csv(out / "hygiene_table.csv")
flags = flag_outliers(trials, table)
(out / "flags.txt").write_text("\n".join(flags) + "\n" if flags else "no outlier trials or channels\n")
pd.DataFrame([peak_event(x, fs) for x in trials],
             columns=["channel", "time_s", "value", "n_channels_over_5sd"]) \
  .rename_axis("trial").to_csv(out / "peak_events.csv")
print(len(flags), "flags")

# --- spectra ---
f, P = psd_all(trials, fs)
plt.figure(figsize=(7, 4))
for t in range(n): plt.loglog(f[1:], P[t, 1:], lw=0.7, alpha=0.6)
plt.xlabel(f"frequency (Hz, assumed fs={args.fs:g})"); plt.ylabel("PSD (µV²/Hz)")
plt.title(f"Channel-averaged PSD, {n} trials"); plt.grid(alpha=0.3, which="both")
plt.savefig(out / "psd.png", dpi=150, bbox_inches="tight"); plt.close()

peaks = pd.DataFrame([fit_peak(f, P[t]) for t in range(n)],
                     columns=["f_peak_Hz", "FWHM_Hz", "prominence_log10"])
peaks.rename_axis("trial").to_csv(out / "peaks.csv")
fpk = float(peaks.f_peak_Hz.median())
M = choose_M(f, P.mean(axis=0), fs)
pd.Series(band_power_rel(f, P), name="hf_power_rel").sort_values(ascending=False) \
  .rename_axis("trial").to_csv(out / "hf_power_rank.csv")
bc = [bimodality_coefficient(envelope(x[args.env_channel], fs, fpk)) for x in trials]
pd.Series(bc, name="bimodality_coeff").rename_axis("trial").to_csv(out / "envelope_bimodality.csv")

# --- stationarity, top-d PCs of all trials pooled ---
exclude = parse_exclude(args.exclude)
Xp = interpolate_channels(trials, exclude)
_, V_all = pca_basis(Xp)
Z = [V_all[:, :args.d].T @ x for x in Xp]
sweep = pd.DataFrame(stationarity_sweep(Z, fs), columns=["win_s", "within_med", "between_med", "ratio"])
sweep.to_csv(out / "stationarity_sweep.csv", index=False)
plt.figure(figsize=(6, 4))
plt.semilogx(sweep.win_s, sweep.ratio, "o-"); plt.axhline(1.0, color="gray", ls="--", lw=0.8)
plt.xlabel("window length (s)"); plt.ylabel("within / between covariance distance")
plt.savefig(out / "stationarity_sweep.png", dpi=150, bbox_inches="tight"); plt.close()

win = int(10 * fs); nc = int(np.ceil(np.sqrt(n)))
fig, axes = plt.subplots(nc, nc, figsize=(13, 9), squeeze=False)
for t, ax in enumerate(axes.ravel()):
    if t >= n: ax.axis("off"); continue
    for name, (lo, hi) in BANDS.items():
        ax.plot(band_power_series(trials[t], fs, win, lo, hi), lw=0.8, label=name)
    ax.set_title(f"trial {t}", fontsize=8); ax.tick_params(labelsize=6)
axes[0, 0].legend(fontsize=6)
fig.suptitle("Band power per 10 s window — fluctuation vs. drift"); fig.tight_layout()
fig.savefig(out / "band_power.png", dpi=150); plt.close(fig)
pd.DataFrame(drift_tstats(trials, fs, BANDS), columns=["trial", "band", "t"]) \
  .to_csv(out / "drift_tstats.csv", index=False)

# --- dimensionality: PCs from the first n_train trials, scored on the rest ---
train, test = Xp[:args.n_train], Xp[args.n_train:]
dims = pd.DataFrame(variance_captured(train, test), columns=["d", "in_sample_%", "held_out_%"])
dims.to_csv(out / "dimensionality.csv", index=False)
plt.figure()
plt.plot(dims.d, dims["in_sample_%"], "o-", label="in-sample")
plt.plot(dims.d, dims["held_out_%"], "s-", label="held-out")
plt.xlabel("d"); plt.ylabel("% variance captured"); plt.legend()
plt.savefig(out / "dimensionality.png", dpi=150, bbox_inches="tight"); plt.close()

_, V = pca_basis(train); proj = V[:, :args.d].T
taus = np.array([[tau_c(zk, fs) for zk in proj @ x] for x in Xp])
neff = np.array([x.shape[1]/fs / tk for x, tk in zip(Xp, taus)])
pd.DataFrame(np.hstack([taus, neff]),
             columns=[f"tau_c_pc{k}" for k in range(args.d)] + [f"n_eff_pc{k}" for k in range(args.d)]) \
  .rename_axis("trial").to_csv(out / "correlation_time.csv")

summary = {"n_trials": n, "fs_assumed": args.fs, "M": M, "fs_after_decimation": fs/M,
           "median_peak_Hz": fpk, "exclude": {str(k): v for k, v in exclude.items()},
           "n_flags": len(flags)}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2)); print("written to", out)
