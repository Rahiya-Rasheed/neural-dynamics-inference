import numpy as np
from .kalman import kalman_filter
from .likelihood import innovation_loglikelihood


def windowed_llr(y, params_pre, params_post, window=50):
    scores = []

    for end in range(window, len(y) + 1):
        block = y[end - window:end]

        f0 = kalman_filter(
            block,
            params_pre["A"],
            params_pre["H"],
            params_pre["Q"],
            params_pre["R"],
        )
        f1 = kalman_filter(
            block,
            params_post["A"],
            params_post["H"],
            params_post["Q"],
            params_post["R"],
        )

        ll0 = innovation_loglikelihood(block, f0, params_pre["H"], params_pre["R"])
        ll1 = innovation_loglikelihood(block, f1, params_post["H"], params_post["R"])
        scores.append(ll1 - ll0)

    return np.asarray(scores)
