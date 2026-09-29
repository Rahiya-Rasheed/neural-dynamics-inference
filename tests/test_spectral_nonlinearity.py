import numpy as np

from src.nonlinearity import (lag_antisymmetry, nonlinear_gain, poly_feats,
                              surrogate_independent, surrogate_joint)
from src.spectral import choose_M, fit_peak, psd_all
from src.simulate import synthetic_trials


def test_peak_found_near_true_rhythm():
    trials = synthetic_trials(n_trials=2, n_channels=8, duration_s=60.0, f_rot=6.0, seed=2)
    f, P = psd_all(trials, 250.0)
    fpk, _, prom = fit_peak(f, P[0])
    assert abs(fpk - 6.0) < 0.5 and prom > 0


def test_choose_M_respects_lowpass():
    trials = synthetic_trials(n_trials=2, n_channels=8, duration_s=30.0, lowpass_hz=20.0, seed=3)
    f, P = psd_all(trials, 250.0)
    M = choose_M(f, P.mean(0), 250.0)
    assert M >= 2 and 250.0 / (2 * M) > 15.0


def test_poly_feats_count():
    z = np.random.default_rng(0).normal(size=(3, 100))
    assert poly_feats(z).shape == (19, 100)      # 3 + 6 + 10 monomials


def test_surrogates_keep_amplitude_spectrum():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(3, 512)).cumsum(axis=1)
    for s in (surrogate_joint(z, rng), surrogate_independent(z, rng)):
        assert np.allclose(np.abs(np.fft.rfft(s, axis=1)), np.abs(np.fft.rfft(z, axis=1)))


def test_joint_surrogate_keeps_cross_spectrum():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(2, 256))
    s = surrogate_joint(z, rng)
    Fz, Fs = np.fft.rfft(z, axis=1), np.fft.rfft(s, axis=1)
    assert np.allclose(Fz[0] * Fz[1].conj(), Fs[0] * Fs[1].conj())


def test_linear_process_has_no_nonlinear_gain_and_rotation_shows_antisymmetry():
    rng = np.random.default_rng(4)
    c, s_ = np.cos(0.3) * 0.95, np.sin(0.3) * 0.95
    A = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 0.9]])
    z = np.zeros((3, 5000))
    for k in range(1, 5000):
        z[:, k] = A @ z[:, k - 1] + rng.normal(size=3)
    assert nonlinear_gain(z) < 0.01
    assert lag_antisymmetry(z, [1])[0] > 0.2
    # a reversible process (symmetric A) has almost none
    zr = np.zeros((3, 5000))
    for k in range(1, 5000):
        zr[:, k] = 0.9 * zr[:, k - 1] + rng.normal(size=3)
    assert lag_antisymmetry(zr, [1])[0] < 0.1
