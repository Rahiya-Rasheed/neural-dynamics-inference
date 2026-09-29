"""Continuous-time view of the fit: Van Loan, pipeline simulation, landscape/rotation split."""
import warnings
import numpy as np
from scipy.linalg import expm, logm, solve_continuous_lyapunov
from scipy.signal import decimate


def van_loan(Ac, Qc, h):
    """Exact discretization: Ad = e^{Ac h}, Qd = ∫₀ʰ e^{Ac s} Qc e^{Ac' s} ds."""
    n = Ac.shape[0]
    Mx = np.zeros((2*n, 2*n))
    Mx[:n, :n] = -Ac; Mx[:n, n:] = Qc; Mx[n:, n:] = Ac.T
    E = expm(Mx * h)
    Ad = E[n:, n:].T
    Qd = Ad @ E[:n, n:]
    return Ad, (Qd + Qd.T)/2


def continuous_noise(Ac, Qd, h, iters=60):
    """Recover Qc from the fitted discrete Qd (fixed-point inversion)."""
    Qc = Qd / h
    for _ in range(iters):
        _, Qd_try = van_loan(Ac, Qc, h)
        Qc = Qc + (Qd - Qd_try)/h
        Qc = (Qc + Qc.T)/2
    return Qc


def discrete_to_continuous(A, Q, dt, imag_tol=1e-6):
    """Ac = logm(A)/dt and matching Qc. Warns if logm(A) is noticeably complex
    (e.g. a negative real eigenvalue) — then the real part is only an approximation."""
    L = logm(A)
    imag = np.abs(np.imag(L)).max()
    if imag > imag_tol * max(1.0, np.abs(L).max()):
        warnings.warn(f"logm(A) has imaginary part {imag:.2e}; continuous drift is approximate")
    Ac = np.real(L)/dt
    return Ac, continuous_noise(Ac, Q, dt)


def psd_sqrt(Q):
    """Square root that tolerates singular Q (rank-deficient by construction:
    only the 'current' state dims receive noise)."""
    w, V = np.linalg.eigh((Q + Q.T)/2)
    return V @ np.diag(np.sqrt(np.clip(w, 0, None)))


def simulate_through_pipeline(Ac, Qc, H, R, M, fs, n_out, rng):
    """Simulate at the original rate, then decimate by M like the real data."""
    Ad, Qd = van_loan(Ac, Qc, 1.0/fs)
    Lq = psd_sqrt(Qd)
    dx, dobs = Ac.shape[0], H.shape[0]
    T = n_out * M
    x = np.zeros(dx); Y = np.zeros((dobs, T))
    for k in range(T):
        x = Ad @ x + Lq @ rng.standard_normal(dx)
        # R is the variance AFTER decimation; the FIR filter averages ~M samples,
        # so pre-decimation white noise needs variance R*M
        Y[:, k] = H @ x + rng.normal(0, np.sqrt(R * M))
    return Y if M == 1 else decimate(Y, M, axis=1, ftype="fir", zero_phase=True)


def landscape_decomposition(Ac, Dc):
    """Pi solves Ac Pi + Pi Ac' = -Dc (the landscape). Ac Pi = S + B, S symmetric,
    B antisymmetric. S = -Dc/2 by the Lyapunov equation, so B is all the rotation."""
    Pi = solve_continuous_lyapunov(Ac, -Dc)
    S = (Ac@Pi + Pi@Ac.T)/2
    B = (Ac@Pi - Pi@Ac.T)/2
    return {"Pi": Pi, "S": S, "B": B,
            "lyapunov_residual": np.abs(Ac@Pi + Pi@Ac.T + Dc).max(),
            "rotation_ratio": np.linalg.norm(B)/np.linalg.norm(Dc/2)}
