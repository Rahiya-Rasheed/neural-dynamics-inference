# neural-dynamics-inference

Analysis pipeline for multichannel EEG. It looks at the hidden state of a dynamic brain signal from noisy, partially observed recordings: how many dimensions the state has, whether it changes with time, and whether a linear Gaussian model is enough to describe it. The goal is not just a model that fits, but an understanding of why it fits, which assumptions it depends on, and what happens when those assumptions are not perfect.

Model:

$$
x_{k+1} = A x_k + w_k,\qquad y_k = H x_k + v_k,
$$

fit by Kalman EM, then read in continuous time as

$$
dx_t = A_c x_t\,dt + dW_t,\qquad \mathrm{Cov}(dW_t) = D\,dt.
$$

## Questions

1. Is the data clean enough to model, and which trials or channels need a decision?
2. Is there one time-invariant state, or several regimes?
3. How many latent dimensions hold up on held-out trials?
4. Is a linear Gaussian latent model enough, or is there nonlinearity or irreversibility it misses?
5. How does the fitted drift split into landscape (stationary covariance), dissipation and rotation, and how stable is that across trials?

## Data

The EEG recordings this pipeline was built for will not be released. The code is public; the data is not.

To run and test the code, use the synthetic dataset. `make_synthetic_data.py` generates it in the same format as the real recordings, from a known latent SDE (a 6 Hz damped rotation plus a slow coordinate). The true parameters are known, so the fit can be checked against them.

The scripts read a `.mat` file with a 1 × n cell array `Xcell` of (channels × samples) arrays; any recording in that format works. Sampling rate defaults to 250 Hz (`--fs`). Every frequency and decay rate in the output scales with it.

## Pipeline

**`characterize.py`**: hygiene table and outlier flags, per-trial peak events, channel-averaged spectra, dominant rhythm (peak above a 1/f background), decimation factor, envelope bimodality, stationarity sweep (within- vs between-trial covariance distance as the window grows), band-power drift, held-out variance by dimension, correlation times.

**`nonlinearity.py`**: two tests on the decimated latents against phase-randomized surrogates. Does a cubic one-step predictor beat a linear one out of sample? Are lagged cross-covariances antisymmetric (rotation) beyond what independent-phase surrogates give?

**`fit_lds.py`**: VAR(p) in companion form as the starting point, then Kalman EM on the training trials. Reports modes and held-out log-likelihood. Checks lag-1 innovation correlation on real held-out trials against data simulated at the original rate and passed through the same decimation. Converts to continuous time (Van Loan), solves the Lyapunov equation for the stationary covariance, splits the drift into symmetric and antisymmetric parts, and refits each trial separately to check how consistent the rhythm is.

Design choices:

- Flags never delete anything. Channel exclusions are passed on the command line (`--exclude trial:channel`) and interpolated from the other channels, so every exclusion is on record.
- Decimation is anti-aliased. The factor is the largest one leaving less than 0.1% of power above the new Nyquist.
- PCs and model parameters are fit on the first 8 trials and scored on the last 8.

## Quick start

```bash
python -m pip install -r requirements.txt
pytest -q

python experiments/make_synthetic_data.py --out data/synthetic_eeg.mat
python experiments/characterize.py --mat data/synthetic_eeg.mat
python experiments/nonlinearity.py --mat data/synthetic_eeg.mat
python experiments/fit_lds.py --mat data/synthetic_eeg.mat --lags 3
```

Outputs (CSV, PNG, `summary.json`) go to `results/`, which is not tracked.

## Structure

```text
neural-dynamics-inference/
├── src/
│   ├── io.py              loading, saving, anti-aliased decimation
│   ├── hygiene.py         summary table, outlier flags, peak events
│   ├── spectral.py        PSD, rhythm peak, decimation factor, envelope
│   ├── projection.py      channel repair, PCA, held-out variance, correlation time
│   ├── stationarity.py    covariance sweep, band-power drift
│   ├── nonlinearity.py    polynomial gain, surrogates, lag antisymmetry
│   ├── lds.py             VAR init, Kalman filter/smoother, EM, modes
│   ├── continuous.py      Van Loan, continuous noise, pipeline simulation, Lyapunov split
│   ├── simulate.py        synthetic trials with known parameters
│   └── pipeline.py        shared load -> project -> decimate step
├── experiments/
│   ├── make_synthetic_data.py
│   ├── characterize.py
│   ├── nonlinearity.py
│   └── fit_lds.py
├── tests/
└── README.md
```

## Next

How uncertainty in the learned model affects the ability to detect and respond to changes in the hidden state, and which errors in the estimated drift and noise actually change the conclusions (the rotation, the landscape, a detector built on the model).
