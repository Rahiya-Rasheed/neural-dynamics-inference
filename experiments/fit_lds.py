"""Fit and check the latent linear model.

VAR(p) -> Kalman EM on training trials -> modes, held-out LL ->
innovation check against data simulated through the same decimation ->
landscape / rotation split -> per-trial refits.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuous import discrete_to_continuous, landscape_decomposition, simulate_through_pipeline
from src.lds import (companion_init, discrete_modes, dominant_rhythm, innovation_lag1_corr,
                     kalman_em, loglik_per_sample)
from src.pipeline import parse_exclude, prepare_latents

ap = argparse.ArgumentParser()
ap.add_argument("--mat", required=True)
ap.add_argument("--out", default="results/fit_lds")
ap.add_argument("--fs", type=float, default=250.0)
ap.add_argument("--exclude", nargs="*", default=[])
ap.add_argument("--M", type=int, default=None)
ap.add_argument("--lags", type=int, default=3, help="VAR order p, state dim = 3p")
ap.add_argument("--n-iter", type=int, default=25)
ap.add_argument("--n-iter-trial", type=int, default=8)
ap.add_argument("--n-sim", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
prep = prepare_latents(args.mat, args.fs, parse_exclude(args.exclude), M=args.M)
Zd, dt, M, fs, nt = prep["Zd"], prep["dt"], prep["M"], prep["cfg"].fs, prep["n_train"]
train, test = Zd[:nt], Zd[nt:]

# --- fit ---
A0, Q0, H, R0 = companion_init(train, args.lags)
A, Q, R, LLs = kalman_em(train, A0, Q0, R0, H, n_iter=args.n_iter)
freq, decay = discrete_modes(A, dt)
modes = pd.DataFrame({"freq_Hz": freq, "decay_per_s": decay}).sort_values("freq_Hz")
modes.to_csv(out / "modes.csv", index=False)
ll_test = float(np.mean([loglik_per_sample(z, A, Q, R, H) for z in test]))
np.savez(out / "fit.npz", A=A, Q=Q, R=R, H=H, LLs=LLs, dt=dt, M=M)
print(modes.to_string(index=False)); print("held-out per-sample LL:", round(ll_test, 3))

# --- pipeline-aware innovation check: simulate at the ORIGINAL rate, decimate like the data ---
Ac, Dc = discrete_to_continuous(A, Q, dt)
r1_real = np.array([innovation_lag1_corr(z, A, Q, R, H) for z in test])
r1_sim  = np.array([innovation_lag1_corr(simulate_through_pipeline(Ac, Dc, H, R, M, fs, Zd[0].shape[1], rng),
                                         A, Q, R, H) for _ in range(args.n_sim)])

# --- landscape / rotation ---
dec = landscape_decomposition(Ac, Dc)
f_joint, decay_joint = dominant_rhythm(A, dt)

# --- per-trial refits, init from joint fit; track the mode nearest the joint rhythm ---
rows = []
for t, z in enumerate(Zd):
    At, _, _, _ = kalman_em([z], A.copy(), Q.copy(), R.copy(), H, n_iter=args.n_iter_trial)
    ft, dct = discrete_modes(At, dt)
    j = np.argmin(np.abs(ft - f_joint))
    rows.append([t, ft[j], dct[j]])
per_trial = pd.DataFrame(rows, columns=["trial", "rhythm_Hz", "decay_per_s"])
per_trial.to_csv(out / "per_trial_modes.csv", index=False)

ms = lambda s: [float(s.mean()), float(s.std())]
summary = {"M": M, "dt": dt, "lags": args.lags, "state_dim": A.shape[0],
           "em_loglik_first_last": [float(LLs[0]), float(LLs[-1])],
           "heldout_ll_per_sample": ll_test,
           "innov_lag1_real_median": np.median(r1_real, 0).round(4).tolist(),
           "innov_lag1_sim_median":  np.median(r1_sim, 0).round(4).tolist(),
           "lyapunov_residual": float(dec["lyapunov_residual"]),
           "rotation_over_dissipation": float(dec["rotation_ratio"]),
           "dominant_rhythm_Hz": float(f_joint), "dominant_rhythm_decay_per_s": float(decay_joint),
           "per_trial_rhythm_mean_sd": ms(per_trial.rhythm_Hz),
           "per_trial_decay_mean_sd": ms(per_trial.decay_per_s)}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
