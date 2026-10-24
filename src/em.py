import numpy as np


def gaussian_pdf(x, mu, var):
    """Density of x under N(mu, var). Works on scalars and arrays."""
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
