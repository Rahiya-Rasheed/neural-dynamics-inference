"""Nonlinearity and irreversibility, against phase-randomized surrogates."""
from itertools import combinations_with_replacement
import numpy as np


def poly_feats(z, degree=3):
    """All monomials of the rows of z up to degree, standardized. 19 features for d=3."""
    rows = [np.prod(z[list(c)], axis=0)
            for k in range(1, degree + 1)
            for c in combinations_with_replacement(range(z.shape[0]), k)]
    F = np.vstack(rows)
    return (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-12)


def nonlinear_gain(z, lam=1e-3, degree=3):
    """Held-out gain of a polynomial one-step predictor over linear. Last 20% is test."""
    s = int(0.8 * z.shape[1])
    Ytr, Yte = z[:, 1:s], z[:, s+1:]
    def mse(F):
        Ftr, Fte = F[:, :s-1], F[:, s:-1]
        G = Ftr @ Ftr.T
        W = Ytr @ Ftr.T @ np.linalg.inv(G + lam*np.trace(G)/G.shape[0]*np.eye(G.shape[0]))
        return np.mean((Yte - W @ Fte)**2)
    return 1 - mse(poly_feats(z, degree)) / mse(z)


def surrogate_joint(z, rng):
    """Phase-randomized: same spectrum and cross-spectra, linear Gaussian."""
    T = z.shape[1]
    F = np.fft.rfft(z, axis=1)
    ph = np.exp(1j * rng.uniform(0, 2*np.pi, F.shape[1])); ph[0] = 1
    if T % 2 == 0: ph[-1] = 1
    return np.fft.irfft(F * ph[None, :], n=T, axis=1)


def surrogate_independent(z, rng):
    """Independent phases per channel: kills cross-phase -> null for antisymmetry."""
    T = z.shape[1]
    F = np.fft.rfft(z, axis=1)
    ph = np.exp(1j * rng.uniform(0, 2*np.pi, F.shape)); ph[:, 0] = 1
    if T % 2 == 0: ph[:, -1] = 1
    return np.fft.irfft(F * ph, n=T, axis=1)


def lag_antisymmetry(z, lags):
    """||C_L - C_L'|| / ||C_L + C_L'||. Zero for a reversible process; rotation makes it > 0."""
    zc = z - z.mean(1, keepdims=True)
    out = []
    for L in lags:
        Cl = zc[:, :-L] @ zc[:, L:].T / (z.shape[1] - L)
        out.append(np.linalg.norm(Cl - Cl.T) / np.linalg.norm(Cl + Cl.T))
    return np.array(out)
