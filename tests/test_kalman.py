import numpy as np

from src.kalman import kalman_filter, rts_smoother
from src.simulate import simulate_lds


def test_filter_shapes():
    z, y, p = simulate_lds(n_steps=50, latent_dim=2, obs_dim=5)
    result = kalman_filter(y, p["A0"], p["H"], p["Q"], p["R"])

    assert result["filtered_mean"].shape == z.shape
    assert result["filtered_cov"].shape == (50, 2, 2)


def test_smoother_shapes():
    z, y, p = simulate_lds(n_steps=50, latent_dim=2, obs_dim=5)
    result = kalman_filter(y, p["A0"], p["H"], p["Q"], p["R"])
    smooth = rts_smoother(result, p["A0"])

    assert smooth["smoothed_mean"].shape == z.shape
    assert np.isfinite(smooth["smoothed_mean"]).all()
