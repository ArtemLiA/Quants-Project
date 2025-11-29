import numpy as np


def generate_multivariate_normal(mean, cov, size):
    n = len(mean)
    L = np.linalg.cholesky(cov)
    z = np.random.randn(*size, n)
    return mean + np.dot(z, L.T)
