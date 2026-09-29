"""Shared preprocessing: load -> choose M -> repair channels -> project -> decimate."""
from .io import Config, load_trials
from .projection import interpolate_channels, pca_basis, project_decimate
from .spectral import choose_M, psd_all


def parse_exclude(items):
    """['8:46', '3:1,2'] -> {8: [46], 3: [1, 2]}"""
    out = {}
    for item in items or []:
        t, chans = item.split(":")
        out.setdefault(int(t), []).extend(int(c) for c in chans.split(","))
    return out


def prepare_latents(mat_path, fs=250.0, exclude=None, d=3, n_train=8, M=None):
    """PCs fit on the first n_train trials only, then applied to all."""
    cfg = Config(mat_path=mat_path, fs_assumed=fs)
    trials = load_trials(cfg)
    if M is None:
        f, P = psd_all(trials, cfg.fs)
        M = choose_M(f, P.mean(axis=0), cfg.fs)
    Xp = interpolate_channels(trials, exclude or {})
    _, V = pca_basis(Xp[:n_train])
    proj = V[:, :d].T
    return {"cfg": cfg, "trials": trials, "Xp": Xp, "M": M, "proj": proj,
            "Zd": project_decimate(Xp, proj, M), "dt": M/cfg.fs, "n_train": n_train}
