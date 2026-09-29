import numpy as np
import pytest

from src.hygiene import flag_outliers, hygiene_table
from src.io import Config, load_trials, save_trials
from src.simulate import synthetic_trials


def small_trials():
    return synthetic_trials(n_trials=4, n_channels=8, duration_s=4.0, seed=1)


def test_mat_roundtrip(tmp_path):
    trials = small_trials()
    path = tmp_path / "x.mat"
    save_trials(path, trials)
    loaded = load_trials(Config(str(path), n_trials=4, n_channels=8))
    assert all(np.allclose(a, b) for a, b in zip(trials, loaded))


def test_loader_rejects_wrong_channel_count(tmp_path):
    path = tmp_path / "x.mat"
    save_trials(path, small_trials())
    with pytest.raises(ValueError):
        load_trials(Config(str(path), n_trials=4, n_channels=9))


def test_flags_inflated_channel():
    trials = small_trials()
    trials[2][5] *= 10
    flags = flag_outliers(trials, hygiene_table(trials))
    assert any("trial 2, channel 5" in f for f in flags)
