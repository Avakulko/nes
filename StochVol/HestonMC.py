import numpy as np
from time import time

def MC(params, S, nsim=1000, N=24*60):
    # , S, strike, tau, r,

    # N - Number of small sub-steps (time)
    dt = 1 / 365 / 24 / 60  # 1 min time step
    
    # Parameters for Heston process
    # V_0 - Initial variance is square of volatility
    # k - Speed of mean reversion
    # theta - Long-run variance
    # nu - Volatility of volatility
    # rho - Correlation

    # Parameters for Heston process
    V_0, k, theta, nu, rho = params  # Initial variance is square of volatility

    V_t = np.ones((N, nsim))
    S_t = np.ones((N, nsim))

    V_t[0, :] *= V_0
    S_t[0, :] *= S


    # Generate Monte-Carlo paths
    for t in range(1, N):
        # Random numbers for S_t and V_t
        Z_s = np.random.normal(size=nsim)
        Z_v = rho * Z_s + np.sqrt(1 - rho ** 2) * np.random.normal(size=nsim)

        # Euler integration
        V_t[t-1, :] = np.maximum(V_t[t-1, :], 0)
        S_t[t, :] = S_t[t-1, :] * (1 + r * dt + np.sqrt(V_t[t-1, :] * dt) * Z_s)
        # S_t[t, :] = S_t[t-1, :] * np.exp(np.sqrt(V_t[t-1, :] * dt) * Z_s - V_t[t-1, :] * dt / 2)
        V_t[t, :] = V_t[t-1, :] + k * (theta - V_t[t-1, :]) * dt + nu * np.sqrt(V_t[t-1, :] * dt) * Z_v  # Volatility process

    # option_price = np.exp(-r * tau) * np.mean(np.maximum(S_t - strike, 0))

    return S_t, V_t

if __name__ == '__main__':
    np.random.seed(42)

    # params = [sigma_t, k, theta, nu, rho]
    # S, strike, tau, r = 9847.744, 18000, 0.12586375889776902, 0.0152

    params = [ 0.79786259,  5.,          0.4579982,   2.14008933, -0.1862887 ]
    S, strike, tau, r = 4299.68, 5750, 0.0201548679921377, 0.0235

    start = time()
    # print(MC(params, S), f' in {time() - start} seconds')
    S_t, V_t = MC(params, S)

    import matplotlib.pyplot as plt
    plt.plot(S_t)
    plt.show()
