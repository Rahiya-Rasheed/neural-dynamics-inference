"""Is linear Gaussian enough? Polynomial gain vs joint-phase surrogates,
lag antisymmetry vs independent-phase surrogates."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nonlinearity import lag_antisymmetry, nonlinear_gain, surrogate_independent, surrogate_joint
from src.pipeline import parse_exclude, prepare_latents

LAGS = [1, 2, 5, 10, 20]

ap = argparse.ArgumentParser()
ap.add_argument("--mat", required=True)
ap.add_argument("--out", default="results/nonlinearity")
ap.add_argument("--fs", type=float, default=250.0)
ap.add_argument("--exclude", nargs="*", default=[])
ap.add_argument("--M", type=int, default=None, help="override decimation factor")
ap.add_argument("--n-surrogates", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
prep = prepare_latents(args.mat, args.fs, parse_exclude(args.exclude), M=args.M)
Zd = prep["Zd"]; n = len(Zd)

g_real = np.array([nonlinear_gain(z) for z in Zd])
g_sur  = np.array([nonlinear_gain(surrogate_joint(Zd[i % n], rng)) for i in range(args.n_surrogates)])
a_real = np.array([lag_antisymmetry(z, LAGS) for z in Zd])
a_null = np.array([lag_antisymmetry(surrogate_independent(z, rng), LAGS) for z in Zd])

rng_ = lambda g: {"median": float(np.median(g)), "min": float(g.min()), "max": float(g.max())}
res = {"M": prep["M"], "gain_real": rng_(g_real), "gain_surrogate": rng_(g_sur), "lags": LAGS,
       "antisymmetry_real_median": np.median(a_real, 0).round(4).tolist(),
       "antisymmetry_null_median": np.median(a_null, 0).round(4).tolist()}
np.savez(out / "nonlinearity.npz", g_real=g_real, g_sur=g_sur, a_real=a_real, a_null=a_null, lags=LAGS)
(out / "summary.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
