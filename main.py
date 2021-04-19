import numpy as np
import pandas as pd
from Heston import C_Heston
from HestonMC import MC
from HestonCalibration import calibrate
from datetime import datetime

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

    with open(filename, 'a') as f:
        log.to_csv(f, header=f.tell() == 0, index=False)


if __name__ == '__main__':
    data = pd.read_csv('Data/data.csv')
    filename = 'Out/log.csv'


    data.date = pd.to_datetime(data.date)
    dates = data.date.unique()
    t = dates[500]
    data = data[data.date < t]
    # data = data[data.type == 'call']

    # method = 'dif_ev'
    method = 'shgo'
    weights = True

    res, calibration_time = calibrate(data, method=method, weights=weights)
    params = res.x



    data['C_Heston_opt'] = data.apply(lambda x: C_Heston(params, x['index_price'], x['strike'], x['tt'], x['irate']),
                                      axis=1)
    write_log()
    # mc = list()
    # for i in range(len(data)):
    #     print(i / len(data) * 100)
    #     mc.append(
    #         MC(params, data.iloc[i]['index_price'], data.iloc[i]['strike'], data.iloc[i]['tt'],
    #            data.iloc[i]['irate']))
    # data['MC'] = mc

    data.to_csv('Out/output.csv', index=False)
    pass
