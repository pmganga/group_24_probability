import numpy as np


def gaussian_pdf(x, mu, var):
    """Density of x under N(mu, var). Works on scalars and arrays."""
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


def weighted_densities(x, params):
    """pi_k * N(x | mu_k, var_k) for each component. Shape (n, 2)."""
    x = np.atleast_1d(x)
    return np.column_stack([
        params['pi'][k] * gaussian_pdf(x, params['mu'][k], params['var'][k])
        for k in range(2)
    ])


def initialise(data):
    """Start the means at the 25th and 75th percentiles so the components differ."""
    var = np.var(data)
    return {'mu':  np.array([np.percentile(data, 25), np.percentile(data, 75)]),
            'var': np.array([var, var]),
            'pi':  np.array([0.5, 0.5])}


def e_step(data, params):
    """Responsibilities: P(component k | x_i). Each row sums to 1."""
    w = weighted_densities(data, params)
    return w / w.sum(axis=1, keepdims=True)


def m_step(data, resp):
    """Re-estimate parameters, weighting each point by its responsibility."""
    n_k = resp.sum(axis=0)
    mu = (resp * data[:, None]).sum(axis=0) / n_k
    var = (resp * (data[:, None] - mu) ** 2).sum(axis=0) / n_k
    return {'mu': mu, 'var': var, 'pi': n_k / len(data)}
