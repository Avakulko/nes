from scipy.stats import norm
import numpy as np
from HestonMC import MC


def C_Heston(params, spot, strike, tau, r):
    sigma_t, k, theta, nu, rho = params

    x = np.log(spot)
    x_star = np.log(strike) - r * tau
    D_t = np.exp(-k * tau)
    v_t = np.sqrt(theta + (sigma_t ** 2 - theta) / (k * tau) * (1 - D_t))
    d_plus = (x - x_star) / (v_t * np.sqrt(tau)) + 1 / 2 * v_t * np.sqrt(tau)
    d_minus = (x - x_star) / (v_t * np.sqrt(tau)) - 1 / 2 * v_t * np.sqrt(tau)
    BS = np.exp(x) * norm.cdf(d_plus) - strike * np.exp(-r * tau) * norm.cdf(d_minus)
    # print(f'BS = {BS}')

    R_t = nu ** 2 / (8 * k ** 2) * (
            theta * tau + (sigma_t ** 2 - theta) / k * (1 - D_t) - 2 * theta / k * (1 - D_t) - 2 * (
            sigma_t ** 2 - theta) * tau * D_t + theta / (2 * k) * (1 - np.exp(-2 * k * tau)) + (
                    sigma_t ** 2 - theta) / k * (D_t - np.exp(-2 * k * tau)))
    K = np.exp(x) / (v_t * np.sqrt(2 * np.pi * tau)) * np.exp(-1 / 2 * d_plus ** 2) * (
            (d_plus ** 2 - v_t * d_plus * np.sqrt(tau) - 1) / (tau * v_t ** 2))

    U_t = rho * nu / (2 * k ** 2) * (
            theta * k * tau - 2 * theta + sigma_t ** 2 + D_t * (2 * theta - sigma_t ** 2) - k * tau * D_t * (
            sigma_t ** 2 - theta))
    H = np.exp(x) / (v_t * np.sqrt(2 * np.pi * tau)) * np.exp(-1 / 2 * d_plus ** 2) * (
            1 - d_plus / (v_t * np.sqrt(tau)))

    return BS + H * U_t + K * R_t


if __name__ == '__main__':
    S = 400
    strike = 250
    r = 0.03
    tau = 0.50137

    sigma_t = 0.1197  # Variance
    k = 1.98937
    theta = 0.3 ** 2
    nu = 0.33147
    rho = -0.45648749

    params = [sigma_t, k, theta, nu, rho]
    print(f'C_Heston = {C_Heston(params, S, strike, tau, r)}')
    print(f'MC = {MC(params, S, strike, tau, r)}')
    # print(f'BS = {BS(tau, x, np.sqrt(v_t), strike, r)}')
    pass
