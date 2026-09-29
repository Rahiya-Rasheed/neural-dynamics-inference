import numpy as np


def fit_pca(x, n_components):
    x_mean = x.mean(axis=0, keepdims=True)
    x_centered = x - x_mean

    _, _, vh = np.linalg.svd(x_centered, full_matrices=False)
    components = vh[:n_components]
    return x_mean, components


def transform_pca(x, mean, components):
    return (x - mean) @ components.T
