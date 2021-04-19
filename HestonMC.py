import random
import numpy as np


def MC(params, S_0, K, T, r):
    # Montecarlo simulation
    random.seed(5000)  # Set the random seed
    N = 1000  # Number of small sub-steps (time)
    n = 100000  # Number of Monte carlo paths

    dt = T / N  # No. of Time step

    # Parameters for Heston process
    V_0 = params[0] ** 1  # Initial variance is square of volatility
    variance = params[0]  # Initial variance is square of volatility
    kappa = params[1]  # Speed of mean reversion
    theta = params[2]  # Long-run variance
    epsilon = params[3]  # Volatility of volatility
    rho = params[4]  # Correlation

    # Integrate equations: Euler method, Montecarlo vectorized
    V_t = np.ones(n) * V_0
    S_t = np.ones(n) * S_0

    # # Generate Montecarlo paths
    # for t in range(1, N):
    #     # Random numbers for S_t and V_t
    #     Z_s = np.random.normal(size=n)
    #     Z_v = rho * Z_s + np.sqrt(1 - rho ** 2) * np.random.normal(size=n)
    #
    #     # Euler integration
    #     V_t = np.maximum(V_t, 0)
    #     S_t = S_t * np.exp(np.sqrt(V_t * dt) * Z_s - V_t * dt / 2)  # Stock price process
    #     V_t = V_t + kappa * (theta - V_t) * dt + epsilon * np.sqrt(V_t * dt) * Z_v  # Volatility process

    # Generate Montecarlo paths
    for t in range(1, N):
        # Random numbers for S_t and V_t
        Z_s = np.random.normal(size=n)
        Z_v = rho * Z_s + np.sqrt(1 - rho ** 2) * np.random.normal(size=n)

        # Euler integration
        V_t = np.maximum(V_t, 0)
        S_t = S_t + r * S_t * dt + S_t * np.sqrt(V_t * dt) * Z_s
        V_t = V_t + kappa * (theta - V_t) * dt + epsilon * np.sqrt(V_t * dt) * Z_v  # Volatility process

    option_price = np.mean(np.exp(-r * T) * np.maximum(S_t - K, 0))

    return option_price

if __name__ == '__main__':
    sigma_t = 0.41  # Variance
    k = 2.08
    theta = np.sqrt(0.93) ** 2
    nu = 2.36
    rho = -0.09
    params = [sigma_t, k, theta, nu, rho]
    S_0, K, T, r = 9847.744, 18000, 0.12586375889776902, 0.0152
    print(MC(params, S_0, K, T, r))