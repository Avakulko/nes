import os
import numpy as np
import pandas as pd
from StochVol.HestonCalibration import calibrate_Heston
from JumpDiffusion.MertonCalibration import calibrate_Merton

np.random.seed(42)
os.chdir('/Users/a17072452/Documents/GitHub/nes')

if __name__ == '__main__':
    data = pd.read_csv('Data/data.csv')

    data.date = pd.to_datetime(data.date)
    dates = data.date.unique()
    # data = data[data.type == 'call']

    t = dates[700]
    data = data[data.date == t]
    data['irate'] = np.mean(data['irate'])
    data['index_price'] = np.mean(data['index_price'])
    data = data[(data['index_price'] / data['strike'] < 1.5) & (data['index_price'] / data['strike'] > 0.5)]
    data = data[data['tt'] > 0.1]
    data = data.groupby(['strike', 'time_expire']).mean()
    data = data.reset_index()
    # data = data[['strike', 'tt', 'index_price', 'irate', 'C_market']]

    weights = False

    if weights:
        w = data['amount'] ** 2
    else:
        w = 1.0

    # calibrate_Heston(data, weights)
    calibrate_Merton(data, weights)

    # start = time()
    # mc_list = list()
    # for i in range(len(data)):
    #     x = data.iloc[i]
    #     mc_list.append(MC(params,
    #                     x['index_price'],
    #                     x['strike'],
    #                     x['tt'],
    #                     x['irate'],
    #                     nsim=10000,
    #                     N=5000))
    # data['MC'] = mc_list
    # print(f'MC time {time() - start}')

    # print(f"{mape(data['C_market'], data['fHes_opt'])}")

    with open('Out/output.csv', 'w') as f:
        data.to_csv(f, index=False)
    pass
