import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from forecasting_metrics import mape, mse
from JumpDiffusion.Merton import merton_jump_call


def write_log(res, C_market, data, t):

    summary = {'t': pd.to_datetime(t).date(),
               'mape': mape(C_market, data['CMerton_opt']),
               'sigma': res.x[0],
               'm': res.x[1],
               'v': res.x[2],
               'lam': res.x[3],
               'success': res.success}

    with open('Out/log_Merton.csv', 'a') as f:
        pd.DataFrame(summary, index=[0]).to_csv(f, header=f.tell() == 0, index=False)


def Merton_obj_function(params, index_price, strike, tt, irate, C_market, weights):
    return mape(merton_jump_call(params, index_price, strike, tt, irate), C_market)


def calibrate_Merton(data, t, weights):
    index_price = np.array(data['index_price'])
    strike = np.array(data['strike'])
    tt = np.array(data['tt'])
    irate = np.array(data['irate'])
    C_market = np.array(data['C_market'])
    args = (index_price, strike, tt, irate, C_market, weights)

    start = time.time()

    fun_min = np.inf
    res_opt = None
    for _ in range(30):
        sigma0 = np.random.uniform(1e-4, 5.0)
        m0 = np.random.uniform(1e-4, 3.0)
        v0 = np.random.uniform(1e-4, 5.0)
        lam0 = np.random.uniform(1e-4, 5.0)
        x0 = np.array([sigma0, m0, v0, lam0])
        res = minimize(fun=Merton_obj_function,
                       x0=x0,
                       bounds=[(1e-8, np.inf), (1e-8, 3.0), (1e-8, np.inf), (1e-8, 5.0)],
                       args=args)
        if res.fun < fun_min:
            res_opt = res
            fun_min = res_opt.fun

    if res_opt is None:
        print(f"t={pd.to_datetime(t).date()} NONE")
        return

    data['CMerton_opt'] = merton_jump_call(res_opt.x, index_price, strike, tt, irate)
    print(f"t={pd.to_datetime(t).date()} mape = {mape(C_market, data['CMerton_opt'])}")

    write_log(res_opt, C_market, data, t)


    return res_opt
