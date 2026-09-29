import numpy as np


def gaussian_logpdf(x, mean, cov):
    x = np.atleast_1d(x)
    mean = np.atleast_1d(mean)

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("covariance must be positive definite")

    diff = x - mean
    quad = diff @ np.linalg.solve(cov, diff)
    d = x.shape[0]

    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def innovation_loglikelihood(y, filter_result, H, R):
    mp = filter_result["predicted_mean"]
    Pp = filter_result["predicted_cov"]

    total = 0.0
    for t in range(len(y)):
        mean_y = H @ mp[t]
        cov_y = H @ Pp[t] @ H.T + R
        total += gaussian_logpdf(y[t], mean_y, cov_y)

    return total
