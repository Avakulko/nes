import numpy as np
from scipy.stats import norm
from scipy.special import factorial


def merton_jump_paths(params, S, r, steps=24*60, Npaths=1000):
    sigma, m, v, lam = params
    size = (steps, Npaths)
    dt = 1 / 365 / 24 / 60
    poi_rv = np.multiply(np.random.poisson(lam * dt, size=size),
                         np.random.normal(m, v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r - sigma ** 2 / 2 - lam * (m + v ** 2 * 0.5)) * dt + sigma * np.sqrt(dt) * np.random.normal(size=size)), axis=0)

    return np.exp(geo + poi_rv) * S


N = norm.cdf


def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def merton_jump_call(params, S, K, T, r):
    sigma, m, v, lam = params
    k = np.arange(40).reshape(-1, 1)
    r_k = r - lam * (m - 1) + k * np.log(m) / T
    sigma_k = np.sqrt(sigma ** 2 + (k * v ** 2) / T)
    pv = np.sum(np.exp(-m * lam * T) * (m * lam * T) ** k / factorial(k) * BS_CALL(S, K, T, r_k, sigma_k), axis=0)
    return pv


if __name__ == '__main__':
    S, K, T, r = np.repeat(100, 2), np.repeat(100, 2), np.repeat(1, 2), np.repeat(0.02, 2)
    m = 0  # mean of jump size
    v = 0.3  # standard deviation of jump
    lam = 1  # intensity of jump i.e. number of jumps per annum
    sigma = 0.2  # annual standard deviation , for Weiner process
    params = np.array([sigma, np.exp(m + v ** 2 * 0.5), v, lam])
    steps = 365  # time steps
    Npaths = 200000  # number of paths to simulate
    np.random.seed(42)
    # j = merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths)  # generate jump diffusion paths

    # mcprice = np.maximum(j[-1] - K, 0).mean() * np.exp(-r * T)  # calculate value of call
    cf_price = merton_jump_call(params, S, K, T, r)

    # print('Monte Carlo Merton Price =', mcprice)
    print('Merton Price =', cf_price)
    print('Black Scholes Price =', BS_CALL(S, K, T, r, sigma))
