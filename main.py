import numpy as np
import pandas as pd
from Heston import C_Heston
from HestonMC import MC
from HestonCalibration import calibrate
from datetime import datetime
import os

os.chdir('/Users/a17072452/Documents/GitHub/nes')


def write_log():
    summary = {'fun': res.fun,
               'params': [params],
               'success': res.success,
               'calibr. time': round(calibration_time, 0),
               'time': datetime.now(),
               'method': method,
               't': pd.to_datetime(t).date(),
               'weights': weights,
               'message': res.message
               }
    # feller, obj_func, move to calibration
    log = pd.DataFrame(summary)

    with open('Out/log.csv', 'a') as f:
        log.to_csv(f, header=f.tell() == 0, index=False)


if __name__ == '__main__':
    data = pd.read_csv('Data/data.csv')

    data.date = pd.to_datetime(data.date)
    dates = data.date.unique()
    t = dates[-1]
    data = data[data.date <= t]
    # data = data[data.type == 'call']

    index_price = data['index_price']
    strike = data['strike']
    tt = data['tt']
    irate = data['irate']
    C_market = data['C_market']

    # method = 'dif_ev'
    method = 'shgo'
    weights = False

    if weights:
        w = data['amount']**2
    else:
        w = 1.0

    res, calibration_time = calibrate(index_price,
                                      strike,
                                      tt,
                                      irate,
                                      C_market,
                                      method=method,
                                      weights=w)
    params = res.x

    data['C_Heston_opt'] = C_Heston(params,
                                    index_price,
                                    strike,
                                    tt,
                                    irate)

    write_log()

    with open('Out/output.csv', 'w') as f:
        data.to_csv(f, index=False)
    pass
