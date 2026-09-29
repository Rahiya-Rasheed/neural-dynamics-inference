"""Linear-Gaussian state space: VAR init, Kalman EM, diagnostics.

x_{k+1} = A x_k + w,  w ~ N(0, Q)
y_k     = H x_k + v,  v ~ N(0, diag(R))     R stored as a vector
"""
import numpy as np


def companion_init(Zs, p, r_scale=0.05, lag_noise=1e-4):
    """VAR(p) by least squares, in companion form. Returns A0, Q0, H, R0.
    EM updates all of A afterwards — companion structure is a start, not a constraint."""
    dobs = Zs[0].shape[0]; dx = dobs * p
    X = []; Y = []
    for z in Zs:
        T = z.shape[1]
        X.append(np.vstack([z[:, p-j:T-j] for j in range(1, p+1)])); Y.append(z[:, p:])
    X = np.hstack(X); Y = np.hstack(Y)
    W = Y@X.T@np.linalg.inv(X@X.T)
    Qe = np.cov(Y - W@X)

    A0 = np.zeros((dx,dx)); A0[:dobs,:] = W; A0[dobs:,:dx-dobs] = np.eye(dx-dobs)
    Q0 = np.zeros((dx,dx)); Q0[:dobs,:dobs] = Qe; Q0[dobs:,dobs:] = lag_noise*np.eye(dx-dobs)
    H  = np.zeros((dobs,dx)); H[:,:dobs] = np.eye(dobs)
    R0 = r_scale*np.diag(np.cov(Zs[0]))
    return A0, Q0, H, R0


def kalman_filter(Y, A, Q, R, H):
    """Forward pass. Also returns total LL and whitened innovations (I-covariance
    under the model). First state initialized from Y[:,0], assumes H = [I 0]."""
    dobs, T = Y.shape; dx = A.shape[0]
    xp = np.zeros((T,dx)); Pp = np.zeros((T,dx,dx))
    xf = np.zeros((T,dx)); Pf = np.zeros((T,dx,dx))
    white = np.zeros((T,dobs)); LL = 0.0
    m_ = np.zeros(dx); m_[:dobs] = Y[:,0]; P = np.eye(dx)*np.var(Y)
    for k in range(T):
        if k > 0: m_ = A@xf[k-1]; P = A@Pf[k-1]@A.T + Q
        xp[k] = m_; Pp[k] = P
        S = H@P@H.T + np.diag(R); Sinv = np.linalg.inv(S)
        innov = Y[:,k] - H@m_
        LL += -0.5*(np.linalg.slogdet(S)[1] + innov@Sinv@innov + dobs*np.log(2*np.pi))
        white[k] = np.linalg.cholesky(Sinv).T @ innov
        K = P@H.T@Sinv
        xf[k] = m_ + K@innov; Pf[k] = P - K@H@P
    return {"xp": xp, "Pp": Pp, "xf": xf, "Pf": Pf, "loglik": LL, "white": white}


def rts_smoother(filt, A):
    """Backward pass. Pcs[k+1] = Cov(x_{k+1}, x_k | Y), needed by the M-step."""
    xf, Pf, xp, Pp = filt["xf"], filt["Pf"], filt["xp"], filt["Pp"]
    T, dx = xf.shape
    xs = np.zeros((T,dx)); Ps = np.zeros((T,dx,dx)); Pcs = np.zeros((T,dx,dx))
    xs[-1] = xf[-1]; Ps[-1] = Pf[-1]
    for k in range(T-2, -1, -1):
        J = Pf[k]@A.T@np.linalg.inv(Pp[k+1])
        xs[k] = xf[k] + J@(xs[k+1]-xp[k+1])
        Ps[k] = Pf[k] + J@(Ps[k+1]-Pp[k+1])@J.T
        Pcs[k+1] = Ps[k+1]@J.T
    return xs, Ps, Pcs


def kalman_em(Ys, A, Q, R, H, n_iter=25):
    """EM for A, Q, diag R with H fixed, pooled over trials. LLs[i] = LL before update i."""
    dobs = H.shape[0]; dx = A.shape[0]; LLs = []
    for it in range(n_iter):
        S11 = np.zeros((dx,dx)); S10 = np.zeros((dx,dx)); S00 = np.zeros((dx,dx))
        Sres = np.zeros(dobs); N1 = 0; Nall = 0; LL = 0
        for Y in Ys:
            T = Y.shape[1]
            filt = kalman_filter(Y, A, Q, R, H); LL += filt["loglik"]
            xs, Ps, Pcs = rts_smoother(filt, A)
            Exx = Ps + np.einsum('ti,tj->tij', xs, xs)
            S11 += Exx[1:].sum(0); S00 += Exx[:-1].sum(0)
            S10 += (Pcs[1:] + np.einsum('ti,tj->tij', xs[1:], xs[:-1])).sum(0)
            res = Y.T - xs@H.T
            Sres += (res**2).sum(0) + np.einsum('ij,tjk,ik->i', H, Ps, H)
            N1 += T-1; Nall += T
        A = S10@np.linalg.inv(S00)
        Q = (S11 - A@S10.T - S10@A.T + A@S00@A.T)/N1
        Q = (Q+Q.T)/2 + 1e-10*np.eye(dx)
        R = np.maximum(Sres/Nall, 1e-8)
        LLs.append(LL)
    return A, Q, R, LLs


def loglik_per_sample(Y, A, Q, R, H):
    return kalman_filter(Y, A, Q, R, H)["loglik"] / Y.shape[1]


def innovation_lag1_corr(Y, A, Q, R, H):
    """Lag-1 autocorrelation of each whitened innovation series."""
    e = kalman_filter(Y, A, Q, R, H)["white"][1:]
    return np.array([np.corrcoef(e[:-1,k], e[1:,k])[0,1] for k in range(e.shape[1])])


def discrete_modes(A, dt):
    """Frequency (Hz) and decay (1/s) of each eigenvalue of A."""
    lam = np.log(np.linalg.eigvals(A).astype(complex))/dt
    return np.abs(lam.imag)/(2*np.pi), -lam.real


def dominant_rhythm(A, dt, min_freq=0.1):
    """Least-damped oscillatory mode. Not the max-frequency mode — the extra
    state dims pick up heavily damped junk modes near Nyquist."""
    freq, decay = discrete_modes(A, dt)
    osc = np.where(freq > min_freq)[0]
    if len(osc) == 0: raise ValueError("no oscillatory mode")
    j = osc[np.argmin(decay[osc])]
    return freq[j], decay[j]
