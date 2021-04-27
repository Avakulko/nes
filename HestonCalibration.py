from scipy.optimize import minimize, shgo, differential_evolution, NonlinearConstraint
import numpy as np
from Heston import C_Heston
from forecasting_metrics import mape
import time


def n(params):
    return 4 * params[1] * params[2] / (params[3] ** 2)


def feller_condition(params):
    # 2 k Theta - nu^2 >= 0 => condition is satisfied
    return 2 * params[1] * params[2] - params[3] ** 2  # >=0


def obj_function(params, index_price, strike, tt, irate, C_market, weights):
    # errors = weights * (C_Heston(params, index_price, strike, tt, irate) - C_market)
    # return np.sqrt(np.sum(errors ** 2))
    return mape(C_market, C_Heston(params, index_price, strike, tt, irate))

def callback(xk):
    print(
        f'|| {feller_condition(xk):^+10.3f} | {xk[0]:^+10.3f} | {xk[1]:^+10.3f} | {xk[2]:^+10.3f} | {xk[3]:^+10.3f} | {xk[4]:^+10.3f} ||')


def calibrate(index_price, strike, tt, irate, C_market, method, weights=1.0):
    bounds = [(0.0001, 20.0), (0.0001, 20.0), (0.0001, 20.0), (0.0001, 20.0), (-1.0, 1.0)]
    print(f"|| {'feller':^10} | {'sigma_t':^10} | {'k':^10} | {'theta':^10} | {'nu':^10} | {'rho':^10} ||")

    args = (index_price, strike, tt, irate, C_market, weights)

    start = time.time()
    if method == 'dif_ev':
        constraint = NonlinearConstraint(feller_condition, 0.0, np.inf)
        res = differential_evolution(func=obj_function,
                                     args=args,
                                     bounds=bounds,
                                     constraints=constraint,
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

    calibration_time = time.time() - start
    print(f"|| {'feller':^10} | {'sigma_t':^10} | {'k':^10} | {'theta':^10} | {'nu':^10} | {'rho':^10} ||")
    print(f'Calibration finished in {calibration_time} seconds')

    return res, calibration_time
