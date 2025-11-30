import numpy as np


def generate_multivariate_normal(mean, cov, size):
    n = len(mean)
    L = np.linalg.cholesky(cov)
    z = np.random.randn(*size, n)
    return mean + np.dot(z, L.T)


def get_cir_residuals(
    data: np.ndarray,
    theta: np.ndarray | float,
    alpha: float,
    dt: float,
    normalized: bool = True
):
    """"""
    rt = data[1:]
    rs = data[:-1]

    rs = data[:-1]
    rt = data[1:]
    y = (rt - rs) / np.sqrt(rs)

    if isinstance(theta, np.ndarray):
        theta = theta[:-1]
    
    z = (theta - rs) * dt / np.sqrt(rs)
    residuals = y - alpha * z

    if normalized:
        return residuals / np.std(residuals)
    return residuals


def get_logfx_residuals(
    data_fx: np.ndarray,
    data_rd: np.ndarray,
    data_rf: np.ndarray,
    sigma: float,
    dt: float,
    normalized: bool = True
):
    logfx_t = np.log(data_fx[1:])
    logfx_s = np.log(data_fx[:-1])

    rd_s = data_rd[:-1]
    rf_s = data_rf[:-1]

    y = logfx_t - logfx_s
    z = (rf_s - rd_s) * dt

    residuals = y - z
    if normalized:
        return residuals / np.std(residuals)
    return residuals
