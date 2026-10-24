import csv

import numpy as np


def load_heights(path):
    """Fathers (one per family) and children from the Galton CSV, labels removed.

    Returns (heights, source) where source is only kept to check the fit afterwards.
    """
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    fathers = {r['family']: float(r['father']) for r in rows}  # father repeats on every child's row
    children = [float(r['childHeight']) for r in rows]
    heights = np.array(list(fathers.values()) + children)
    source = np.array(['Father'] * len(fathers) + ['Child'] * len(children))
    return heights, source


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


def log_likelihood(data, params):
    """Sum over points of log(pi_1 * N_1 + pi_2 * N_2)."""
    return np.sum(np.log(weighted_densities(data, params).sum(axis=1)))


def snapshot(params):
    """Flatten params into one row for the tracking table."""
    return {'mu1': params['mu'][0], 'mu2': params['mu'][1],
            'var1': params['var'][0], 'var2': params['var'][1],
            'pi1': params['pi'][0], 'pi2': params['pi'][1]}


def fit_em(data, tol=1e-6, max_iter=1000):
    """Run EM until the log-likelihood stops improving. Returns (params, history)."""
    params = initialise(data)
    history = [{'iteration': 0, **snapshot(params),
                'log_likelihood': log_likelihood(data, params)}]
    for i in range(1, max_iter + 1):
        params = m_step(data, e_step(data, params))
        ll = log_likelihood(data, params)
        history.append({'iteration': i, **snapshot(params), 'log_likelihood': ll})
        if abs(ll - history[-2]['log_likelihood']) < tol:
            break
    return params, history


def tracking_table(history, rows=3, labels=('Children', 'Fathers')):
    """Print the required table for iterations 0..rows-1. Component 1 starts at the lower mean."""
    header = (f"{'Iteration':<10}{'μ1 (' + labels[0] + ')':>16}{'μ2 (' + labels[1] + ')':>16}"
              f"{'σ1²':>10}{'σ2²':>10}{'π1':>9}{'π2':>9}{'Log-Likelihood':>17}")
    print(header)
    print('-' * len(header))
    for h in history[:rows]:
        print(f"{h['iteration']:<10}{h['mu1']:>16.4f}{h['mu2']:>16.4f}{h['var1']:>10.4f}"
              f"{h['var2']:>10.4f}{h['pi1']:>9.4f}{h['pi2']:>9.4f}{h['log_likelihood']:>17.4f}")


def classify(height, params, labels=('Children', 'Fathers')):
    """Print and return P(group | height). labels[0] is the component with the smaller mean."""
    post = e_step(height, params)[0]
    order = np.argsort(params['mu'])
    print(f"Height: {height}")
    for label, k in zip(labels, order):
        print(f"  P({label} | height) = {post[k]:.4f}")
    return dict(zip(labels, post[order]))
