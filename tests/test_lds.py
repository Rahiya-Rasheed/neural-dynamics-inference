import numpy as np

from src.continuous import (continuous_noise, discrete_to_continuous,
                            landscape_decomposition, van_loan)
from src.lds import (companion_init, discrete_modes, kalman_em, kalman_filter)


def simulate(A, Q, H, R, T, rng):
    dx = A.shape[0]
    L = np.linalg.cholesky(Q)
    x = np.zeros(dx); Y = np.zeros((H.shape[0], T))
    for k in range(T):
        x = A @ x + L @ rng.standard_normal(dx)
        Y[:, k] = H @ x + rng.normal(0, np.sqrt(R))
    return Y


def test_scalar_van_loan_matches_ou():
    theta, q, h = 2.0, 3.0, 0.1
    Ad, Qd = van_loan(np.array([[-theta]]), np.array([[q]]), h)
    assert np.isclose(Ad[0, 0], np.exp(-theta * h))
    assert np.isclose(Qd[0, 0], q / (2 * theta) * (1 - np.exp(-2 * theta * h)))


def test_continuous_noise_inverts_van_loan():
    Ac = np.array([[-1.0, -6.0], [6.0, -1.0]])
    Qc = np.array([[2.0, 0.3], [0.3, 1.0]])
    _, Qd = van_loan(Ac, Qc, 0.01)
    assert np.allclose(continuous_noise(Ac, Qd, 0.01), Qc, atol=1e-8)


def test_discrete_to_continuous_roundtrip():
    Ac = np.array([[-2.0, -10.0], [10.0, -2.0]])
    Qc = np.diag([1.0, 0.5])
    Ad, Qd = van_loan(Ac, Qc, 0.02)
    Ac2, Qc2 = discrete_to_continuous(Ad, Qd, 0.02)
    assert np.allclose(Ac2, Ac, atol=1e-8) and np.allclose(Qc2, Qc, atol=1e-6)


def test_landscape_decomposition():
    Ac = np.array([[-2.0, -10.0], [10.0, -2.0]])
    Dc = np.eye(2)
    dec = landscape_decomposition(Ac, Dc)
    assert dec["lyapunov_residual"] < 1e-10
    assert np.allclose(dec["S"], -Dc / 2)
    assert np.allclose(dec["B"], -dec["B"].T) and np.linalg.norm(dec["B"]) > 0


def test_whitened_innovations_have_unit_variance_under_true_model():
    rng = np.random.default_rng(0)
    A = np.array([[0.9, -0.2], [0.2, 0.9]]); Q = 0.1 * np.eye(2)
    H = np.eye(2); R = np.array([0.05, 0.05])
    Y = simulate(A, Q, H, R, 4000, rng)
    w = kalman_filter(Y, A, Q, R, H)["white"][50:]
    assert np.allclose(np.cov(w.T), np.eye(2), atol=0.1)


def test_em_likelihood_does_not_decrease_and_recovers_rotation():
    rng = np.random.default_rng(1)
    th = 0.4
    A = 0.95 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q = 0.1 * np.eye(2); H = np.eye(2); R = np.array([0.02, 0.02])
    Ys = [simulate(A, Q, H, R, 1500, rng) for _ in range(2)]
    A0, Q0, H0, R0 = companion_init(Ys, p=1)
    Af, _, _, LLs = kalman_em(Ys, A0, Q0, R0, H0, n_iter=10)
    assert np.all(np.diff(LLs) > -1e-6 * abs(LLs[0]))
    f_true, _ = discrete_modes(A, 1.0)
    f_fit, _ = discrete_modes(Af, 1.0)
    assert abs(f_fit.max() - f_true.max()) < 0.01


def test_companion_init_shapes():
    rng = np.random.default_rng(2)
    Zs = [rng.normal(size=(3, 300)) for _ in range(2)]
    A0, Q0, H, R0 = companion_init(Zs, p=3)
    assert A0.shape == (9, 9) and Q0.shape == (9, 9) and H.shape == (3, 9) and R0.shape == (3,)
    assert np.allclose(A0[3:, :6], np.eye(6))


def test_dominant_rhythm_ignores_fast_damped_modes():
    from src.lds import dominant_rhythm
    dt = 0.01
    blocks = []
    for f, d in [(6.0, 3.0), (30.0, 60.0)]:
        w = 2 * np.pi * f
        blocks.append(np.array([[-d, -w], [w, -d]]))
    Ac = np.zeros((4, 4)); Ac[:2, :2] = blocks[0]; Ac[2:, 2:] = blocks[1]
    Ad, _ = van_loan(Ac, np.eye(4), dt)
    f, d = dominant_rhythm(Ad, dt)
    assert np.isclose(f, 6.0) and np.isclose(d, 3.0)
