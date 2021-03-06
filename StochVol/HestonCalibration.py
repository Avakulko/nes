import time
import numpy as np
import pandas as pd
from forecasting_metrics import mape
from scipy.optimize import least_squares, minimize, NonlinearConstraint
from StochVol.Heston import C_Heston, fHes, JacHes


def write_log(res, C_market, data, t):
    summary = {'t': pd.to_datetime(t).date(),
               'mape': mape(C_market, data['fHes_opt']),
               'sigma_0': res.x[0],
               'k': res.x[1],
               'theta': res.x[2],
               'nu': res.x[3],
               'rho': res.x[4],
               'success': res.success,
               'feller': feller(res.x)
               }
    # feller
    with open('Out/log_Heston.csv', 'a') as f:
        pd.DataFrame(summary, index=[0]).to_csv(f, header=f.tell() == 0, index=False)


def feller(params):
    # 2 k Theta - nu^2 >= 0 => condition is satisfied
    return 2 * params[1] * params[2] - params[3] ** 2  # >=0


def CHeston_mape(params, index_price, strike, tt, irate, C_market, weights):
    return mape(C_market, C_Heston(params, index_price, strike, tt, irate))


def fHes_mape(params, index_price, strike, tt, irate, C_market, weights):
    return mape(C_market, fHes(params, index_price, strike, tt, irate))


def residuals(params, index_price, strike, tt, irate, C_market, weights):
    return fHes(params, index_price, strike, tt, irate) - C_market


def obj_fun(params, index_price, strike, tt, irate, C_market, weights):
    return np.sum((fHes(params, index_price, strike, tt, irate) - C_market) ** 2)


def calibrate_Heston(data, t, weights):
    index_price = np.array(data['index_price'])
    strike = np.array(data['strike'])
    tt = np.array(data['tt'])
    irate = np.array(data['irate'])
    C_market = np.array(data['C_market'])
    args = (index_price, strike, tt, irate, C_market, weights)

    start = time.time()

    # cost_min = np.inf
    # for _ in range(1):
    #     sigma_00, k0, theta0, nu0 = np.random.uniform(0.0, 5.0, size=4)
    #     rho0 = np.random.uniform(-1.0, 1.0)
    #     x0 = np.array([sigma_00, k0, theta0, nu0, rho0])
    #     print(f'{x0=}')
    #     res = least_squares(fun=residuals,
    #                         jac=JacHes,
    #                         # bounds=[(0.0, 0.0, 0.0, 0.0, -1.0), (10.0, 10.0, 10.0, 10.0, 1.0)],
    #                         x0=x0,
    #                         args=args,
    #                         verbose=2,
    #                         method='lm')
    #     if res.cost < cost_min:
    #         res_opt = res
    #         cost_min = res_opt.cost

    constraint = NonlinearConstraint(feller, 1e-2, np.inf)
    min_fun = np.inf
    for _ in range(10):
        sigma_00, k0, theta0, nu0 = np.random.uniform(1e-2, 5.0, size=4)
        rho0 = np.random.uniform(-1.0, 1.0)
        x0 = np.array([sigma_00, k0, theta0, nu0, rho0])
        print(x0)
        res = minimize(fun=obj_fun,
                       x0=x0,
                       args=args,
                       bounds=[(1e-2, None), (1e-2, None), (1e-2, None), (1e-2, None), (-1.0, 1.0)],
                       constraints=constraint,
                       options={'disp': False})
        if res.fun < min_fun:
            min_fun = res.fun
            res_opt = res

    data['fHes_opt'] = fHes(res_opt.x, index_price, strike, tt, irate)

    # print('OPTIMAL ', feller(res_opt.x), res_opt.x)
    print(f"t={pd.to_datetime(t).date()} mape = {mape(C_market, data['fHes_opt'])}")
    # print(f'Calibration finished in {time.time() - start} seconds')

    write_log(res_opt, C_market, data, t)

    return res_opt


"""

def callback(xk):
    print(
        f'|| {feller_condition(xk):^+10.3f} | {xk[0]:^+10.3f} | {xk[1]:^+10.3f} | {xk[2]:^+10.3f} | {xk[3]:^+10.3f} | {xk[4]:^+10.3f} ||')

def n(params):
    return 4 * params[1] * params[2] / (params[3] ** 2)
    
    
    
    bounds = [(0.0001, 200.0), (0.0001, 200.0), (0.0001, 200.0), (0.0001, 200.0), (-1.0, 1.0)]
    # print(f"|| {'feller':^10} | {'sigma_t':^10} | {'k':^10} | {'theta':^10} | {'nu':^10} | {'rho':^10} ||")

    
    if model == 'dif_ev':
        # constraint = NonlinearConstraint(feller_condition, 0.0, np.inf)
        res = differential_evolution(func=obj_function,
                                     args=args,
                                     # bounds=bounds,
                                     bounds=[(0.0, 5), (0.0, 5), (0.0, 5), (0.0, 5), (-1.0, 1.0)],
                                     # constraints=constraint,
                                     polish=True,
                                     workers=-1)

    if model == 'shgo':
        constraint = {'type': 'ineq',
                      'fun': feller_condition}
        res = shgo(func=obj_function,
                   args=args,
                   bounds=bounds,
                   constraints=constraint,
                   callback=callback,
                   options={'disp': True}
                   )

    if model == 'local':
        # sigma_t, k, theta, nu, rho
        constraint = NonlinearConstraint(feller_condition, 0.1, np.inf)
        min_fun = np.inf
        for _ in range(100):
            sigma_00, k0, theta0, nu0 = np.random.uniform(0.0, 5.0, size=4)
            rho0 = np.random.uniform(-1.0, 1.0)
            x0 = np.array([sigma_00, k0, theta0, nu0, rho0])
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
"""
