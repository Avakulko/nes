import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from forecasting_metrics import mape
from JumpDiffusion.Merton import merton_jump_call


def write_log(res):
    summary = {'fun': res.fun,
               'params': [res.x],
               'success': res.success,
               # 't': pd.to_datetime(t).date(),
               # 'weights': weights,
               # 'message': res.message
               }

    # with open('/Users/a17072452/Documents/GitHub/nes/Out/log_Merton.csv', 'a') as f:
    #     pd.DataFrame(summary).to_csv(f, header=f.tell() == 0, index=False)
    pd.DataFrame(summary).to_csv('/Users/a17072452/Documents/GitHub/nes/Out/log_Merton.csv', header=False, index=False)

def Merton_obj_function(params, index_price, strike, tt, irate, C_market, weights):
    return mape(merton_jump_call(params, index_price, strike, tt, irate), C_market)


def calibrate_Merton(data, weights):
    index_price = np.array(data['index_price'])
    strike = np.array(data['strike'])
    tt = np.array(data['tt'])
    irate = np.array(data['irate'])
    C_market = np.array(data['C_market'])
    options_params = (index_price, strike, tt, irate, C_market, weights)

    start = time.time()

    fun_min = np.inf
    for _ in range(100):
        sigma0 = np.random.uniform(0.0, 10.0)
        m0 = np.random.uniform(0.0, 10.0)
        v0 = np.random.uniform(0.0, 10.0)
        lam0 = np.random.uniform(0.0, 5.0)
        x0 = np.array([sigma0, np.exp(m0 + v0 ** 2 * 0.5), v0, lam0])
        res = minimize(fun=Merton_obj_function,
                       x0=x0,
                       args=options_params)
        if res.fun < fun_min:
            res_opt = res
            fun_min = res_opt.fun

    print(f'Calibration finished in {time.time() - start} seconds')
    write_log(res_opt)

    return res_opt