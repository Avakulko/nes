import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from forecasting_metrics import mape, mse
from JumpDiffusion.Merton import merton_jump_call


def write_log(res):
    summary = {'fun': res.fun,
               'params': [res.x],
               'success': res.success,
               # 't': pd.to_datetime(t).date(),
               # 'weights': weights,
               # 'message': res.message
               }

    with open('Out/log_Merton.csv', 'a') as f:
        pd.DataFrame(summary).to_csv(f, header=f.tell() == 0, index=False)


def Merton_obj_function(params, index_price, strike, tt, irate, C_market, weights):
    return mape(merton_jump_call(params, index_price, strike, tt, irate), C_market)


def calibrate_Merton(data, weights):
    index_price = np.array(data['index_price'])
    strike = np.array(data['strike'])
    tt = np.array(data['tt'])
    irate = np.array(data['irate'])
    C_market = np.array(data['C_market'])
    args = (index_price, strike, tt, irate, C_market, weights)

    start = time.time()

    fun_min = np.inf
    for _ in range(100):
        sigma0 = np.random.uniform(1e-8, 5.0)
        m0 = np.random.uniform(1e-8, 3.0)
        v0 = np.random.uniform(1e-8, 5.0)
        lam0 = np.random.uniform(1e-8, 5.0)
        x0 = np.array([sigma0, m0, v0, lam0])
        res = minimize(fun=Merton_obj_function,
                       x0=x0,
                       bounds=[(1e-8, np.inf), (1e-8, 3.0), (1e-8, np.inf), (1e-8, 5.0)],
                       args=args)
        if res.fun < fun_min:
            res_opt = res
            fun_min = res_opt.fun

    print(f'Calibration finished in {time.time() - start} seconds')
    write_log(res_opt)

    data['CMerton_opt'] = merton_jump_call(res_opt.x, index_price, strike, tt, irate)

    return res_opt
