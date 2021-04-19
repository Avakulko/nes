from scipy.optimize import minimize, shgo, differential_evolution, NonlinearConstraint
import numpy as np
from Heston import C_Heston
import time


def n(params):
    return 4 * params[1] * params[2] / (params[3] ** 2)


def feller_condition(params):
    # nu^2 - 2 k Theta < 0 => condition is satisfied
    return 2 * params[1] * params[2] - params[3] ** 2  # >=0


def obj_function(params, data, weights):
    print('busy...')
    errors = weights * (
            data.apply(lambda x: C_Heston(params, x['index_price'], x['strike'], x['tt'], x['irate']), axis=1) -
            data['C_market']) #/ data['C_market']
    errors = np.array(errors)

    return np.sqrt(np.sum(errors ** 2))

def callback(xk):
    print(f'feller: {round(feller_condition(xk), 5)} sigma_t: {round(xk[0], 5)} k: {round(xk[1], 5)} theta: {round(xk[2], 5)} nu: {round(xk[3], 5)} rho: {round(xk[4], 5)}')
    # return

def calibrate(data, method, weights):
    if weights:
        weights = data['amount']
    else:
        weights = 1

    bounds = [(0.0001, 20), (0.0001, 20), (0.0001, 20), (0.0001, 20), (-1, 1)]
    constraint = NonlinearConstraint(feller_condition, 0, np.inf)
    args = (data, weights)
    start = time.time()
    if method == 'dif_ev':
        res = differential_evolution(func=obj_function,
                                     args=args,
                                     bounds=bounds,
                                     constraints=constraint)

    if method == 'shgo':
        constraint = {'type': 'ineq',
                      'fun': feller_condition}
        res = shgo(func=obj_function,
                   args=args,
                   bounds=bounds,
                   constraints=constraint,
                   callback=callback
                   )


    calibration_time = time.time() - start
    print(f'Calibration finished in {calibration_time} seconds')

    return res, calibration_time
