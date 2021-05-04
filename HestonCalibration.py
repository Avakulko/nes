from scipy.optimize import minimize, shgo, differential_evolution, NonlinearConstraint, brute
import numpy as np
from Heston import C_Heston, fHes
from forecasting_metrics import mape
import time


def n(params):
    return 4 * params[1] * params[2] / (params[3] ** 2)


def feller_condition(params):
    # 2 k Theta - nu^2 >= 0 => condition is satisfied
    return 2 * params[1] * params[2] - params[3] ** 2  # >=0


def obj_function(params, index_price, strike, tt, irate, C_market, weights):
    # return mape(C_market, C_Heston(params, index_price, strike, tt, irate))
    return mape(C_market, fHes(params, index_price, strike, tt, irate))

def callback(xk):
    print(
        f'|| {feller_condition(xk):^+10.3f} | {xk[0]:^+10.3f} | {xk[1]:^+10.3f} | {xk[2]:^+10.3f} | {xk[3]:^+10.3f} | {xk[4]:^+10.3f} ||')


def calibrate(index_price, strike, tt, irate, C_market, method, weights=1.0):
    bounds = [(0.0001, 200.0), (0.0001, 200.0), (0.0001, 200.0), (0.0001, 200.0), (-1.0, 1.0)]
    # print(f"|| {'feller':^10} | {'sigma_t':^10} | {'k':^10} | {'theta':^10} | {'nu':^10} | {'rho':^10} ||")

    args = (index_price, strike, tt, irate, C_market, weights)

    start = time.time()
    if method == 'dif_ev':
        # constraint = NonlinearConstraint(feller_condition, 0.0, np.inf)
        res = differential_evolution(func=obj_function,
                                     args=args,
                                     # bounds=bounds,
                                     bounds=[(0.0, 5), (0.0, 5), (0.0, 5), (0.0, 5), (-1.0, 1.0)],
                                     # constraints=constraint,
                                     polish=True,
                                     workers=-1)

    if method == 'shgo':
        constraint = {'type': 'ineq',
                      'fun': feller_condition}
        res = shgo(func=obj_function,
                   args=args,
                   bounds=bounds,
                   constraints=constraint,
                   callback=callback,
                   options={'disp': True}
                   )

    if method == 'local':
        # sigma_t, k, theta, nu, rho
        constraint = NonlinearConstraint(feller_condition, 0.1, np.inf)
        n_good_iters = 0
        min_fun = 100
        while True:
            sigma_t0, k0, theta0, nu0 = np.random.uniform(0.0, 5.0, size=4)
            rho0 = np.random.uniform(-1.0, 1.0)
            x0 = np.array([sigma_t0, k0, theta0, nu0, rho0])
            if feller_condition(x0) < 0:
                continue
            n_good_iters += 1
            res = minimize(fun=obj_function,
                           x0=x0,
                           args=args,
                           bounds=[(0.0, None), (0.0, None), (0.0, None), (0.0, None), (-1.0, 1.0)],
                           # constraints=constraint,
                           options={'disp': False})
            print(feller_condition(res.x), res.fun, res.x)
            if res.fun < min_fun:
                min_fun = res.fun
                res_opt = res

            if n_good_iters == 1:
                break

    calibration_time = time.time() - start
    # print(f"|| {'feller':^10} | {'sigma_t':^10} | {'k':^10} | {'theta':^10} | {'nu':^10} | {'rho':^10} ||")
    print(f'Calibration finished in {calibration_time} seconds')

    return res_opt, calibration_time
