import numpy as np


def kalman_filter(y, A, H, Q, R, m0=None, P0=None):
    n_steps = y.shape[0]
    latent_dim = A.shape[0]

    m = np.zeros(latent_dim) if m0 is None else np.asarray(m0, dtype=float)
    P = np.eye(latent_dim) if P0 is None else np.asarray(P0, dtype=float)

    filtered_mean = np.zeros((n_steps, latent_dim))
    filtered_cov = np.zeros((n_steps, latent_dim, latent_dim))
    predicted_mean = np.zeros_like(filtered_mean)
    predicted_cov = np.zeros_like(filtered_cov)

    I = np.eye(latent_dim)

    for t in range(n_steps):
        if t == 0:
            m_pred = m
            P_pred = P
        else:
            m_pred = A @ m
            P_pred = A @ P @ A.T + Q

        S = H @ P_pred @ H.T + R
        K = np.linalg.solve(S, H @ P_pred).T

        innovation = y[t] - H @ m_pred
        m = m_pred + K @ innovation
        P = (I - K @ H) @ P_pred

        predicted_mean[t] = m_pred
        predicted_cov[t] = P_pred
        filtered_mean[t] = m
        filtered_cov[t] = P

    return {
        "filtered_mean": filtered_mean,
        "filtered_cov": filtered_cov,
        "predicted_mean": predicted_mean,
        "predicted_cov": predicted_cov,
    }


def rts_smoother(filter_result, A):
    mf = filter_result["filtered_mean"]
    Pf = filter_result["filtered_cov"]
    mp = filter_result["predicted_mean"]
    Pp = filter_result["predicted_cov"]

    ms = mf.copy()
    Ps = Pf.copy()

    for t in range(len(mf) - 2, -1, -1):
        J = np.linalg.solve(Pp[t + 1], A @ Pf[t]).T
        ms[t] = mf[t] + J @ (ms[t + 1] - mp[t + 1])
        Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T

    return {"smoothed_mean": ms, "smoothed_cov": Ps}
