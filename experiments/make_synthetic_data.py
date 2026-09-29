"""Synthetic dataset in the same .mat layout as the recordings."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io import save_trials
from src.simulate import synthetic_trials

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic_eeg.mat")
ap.add_argument("--duration", type=float, default=20.0, help="seconds per trial")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
trials = synthetic_trials(duration_s=args.duration, seed=args.seed)
save_trials(args.out, trials)
print(f"wrote {len(trials)} trials, shape {trials[0].shape} -> {args.out}")
