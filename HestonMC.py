import numpy as np
from time import time

def MC(params, S, strike, tau, r, nsim=10000, N=5000):

    # feller_condition = 2 * params[1] * params[2] - params[3] ** 2
    # if feller_condition < 0:
    #     raise Exception(f'Feller condition = {round(feller_condition, 3)} is not satisfied')

    # N - Number of small sub-steps (time)
    dt = tau / N  # No. of Time step
    
    # Parameters for Heston process
    # V_0 - Initial variance is square of volatility
    # k - Speed of mean reversion
    # theta - Long-run variance
    # nu - Volatility of volatility
    # rho - Correlation

    # Parameters for Heston process
    V_0, k, theta, nu, rho = params  # Initial variance is square of volatility

    # Integrate equations: Euler method, Monte-Carlo vectorized
    V_t = np.ones(nsim) * V_0
    S_t = np.ones(nsim) * S
    
    # antiV_t = np.ones(nsim) * V_0
    # antiS_t = np.ones(nsim) * S

    # Generate Monte-Carlo paths
    for t in range(1, N):
        # Random numbers for S_t and V_t
        Z_s = np.random.normal(size=nsim)
        Z_v = rho * Z_s + np.sqrt(1 - rho ** 2) * np.random.normal(size=nsim)

        # Euler integration
        V_t = np.maximum(V_t, 0)
        S_t *= 1 + r * dt + np.sqrt(V_t * dt) * Z_s
        V_t += k * (theta - V_t) * dt + nu * np.sqrt(V_t * dt) * Z_v  # Volatility process
        # E-M variance
        # V_t += k*(theta-V_t)*dt + nu*np.sqrt(V_t*dt)*Z_v + 1/4*nu**2*(Z_v**2-1)*dt

        # antiV_t = np.maximum(antiV_t, 0)
        # antiS_t *= 1 + r * dt + np.sqrt(antiV_t * dt) * Z_s
        # antiV_t += k * (theta - antiV_t) * dt - nu * np.sqrt(antiV_t * dt) * Z_v
    #
    # S_t = 0.5 * (S_t + antiS_t)
    option_price = np.exp(-r * tau) * np.mean(np.maximum(S_t - strike, 0))

    return option_price

if __name__ == '__main__':
    np.random.seed(42)

    # params = [sigma_t, k, theta, nu, rho]
    # S, strike, tau, r = 9847.744, 18000, 0.12586375889776902, 0.0152

    params = [ 0.79786259,  5.,          0.4579982,   2.14008933, -0.1862887 ]
    S, strike, tau, r = 4299.68, 5750, 0.0201548679921377, 0.0235

    start = time()
    print(MC(params, S, strike, tau, r), f' in {time() - start} seconds')
