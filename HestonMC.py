import numpy as np
from time import time

def MC(params, S, strike, tau, r, nsim=1000):

    feller_condition = 2 * params[1] * params[2] - params[3] ** 2
    if feller_condition < 0:
        raise Exception(f'Feller condition = {round(feller_condition, 3)} is not satisfied')

    # Montecarlo simulation
    # np.random.seed(42)  # Set the random seed
    N = 20000  # Number of small sub-steps (time)

    dt = tau / N  # No. of Time step

    # Parameters for Heston process
    V_0 = params[0] ** 1  # Initial variance is square of volatility
    variance = params[0]  # Initial variance is square of volatility
    kappa = params[1]  # Speed of mean reversion
    theta = params[2]  # Long-run variance
    epsilon = params[3]  # Volatility of volatility
    rho = params[4]  # Correlation

    # Integrate equations: Euler method, Montecarlo vectorized
    V_t = np.ones(nsim) * V_0
    S_t = np.ones(nsim) * S

    # # Generate Montecarlo paths
    # for t in range(1, N):
    #     # Random numbers for S_t and V_t
    #     Z_s = np.random.normal(size=nsim)
    #     Z_v = rho * Z_s + np.sqrt(1 - rho ** 2) * np.random.normal(size=nsim)
    #
    #     # Euler integration
    #     V_t = np.maximum(V_t, 0)
    #     S_t = S_t * np.exp(np.sqrt(V_t * dt) * Z_s - V_t * dt / 2)  # Stock price process
    #     V_t = V_t + kappa * (theta - V_t) * dt + epsilon * np.sqrt(V_t * dt) * Z_v  # Volatility process

    # Generate Montecarlo paths
    for t in range(1, N):
        # Random numbers for S_t and V_t
        Z_s = np.random.normal(size=nsim)
        Z_v = rho * Z_s + np.sqrt(1 - rho ** 2) * np.random.normal(size=nsim)

        # Euler integration
        V_t = np.maximum(V_t, 0)
        S_t *= 1 + r * dt + np.sqrt(V_t * dt) * Z_s
        V_t += kappa * (theta - V_t) * dt + epsilon * np.sqrt(V_t * dt) * Z_v  # Volatility process

    # # Generate Montecarlo paths
    # # Random numbers for S_t and V_t
    # Z_s = np.random.normal(size=(nsim, N))
    # Z_v = rho * Z_s + np.sqrt(1 - rho ** 2) * np.random.normal(size=(nsim, N))
    #
    # # Euler integration
    # V_t = np.maximum(V_t, 0)
    # S_t = S_t + r * S_t * dt + S_t * np.sqrt(V_t * dt) * Z_s
    # V_t = V_t + kappa * (theta - V_t) * dt + epsilon * np.sqrt(V_t * dt) * Z_v  # Volatility process

    option_price = np.exp(-r * tau) * np.mean(np.maximum(S_t - strike, 0))

    return option_price

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
    S, strike, tau, r = 9847.744, 18000, 0.12586375889776902, 0.0152


    nsim = 1000  # Number of Monte carlo paths
    start = time()
    print(MC(params, S, strike, tau, r, nsim=nsim), f' in {time() - start} seconds')
