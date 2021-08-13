import os
import numpy as np
import pandas as pd
from StochVol.HestonCalibration import calibrate_Heston
from JumpDiffusion.MertonCalibration import calibrate_Merton
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
os.chdir('/Users/a17072452/Documents/GitHub/nes')

if __name__ == '__main__':
    data = pd.read_csv('Data/data.csv')

    data.date = pd.to_datetime(data.date)
    dates = data.date.unique()
    # data = data[data.type == 'call']

    # t = dates[700]
    for t in dates:
        day = data[data.date == t]
        day['irate'] = np.mean(day['irate'])
        day['index_price'] = np.mean(day['index_price'])
        day = day[(day['index_price'] / day['strike'] < 1.5) & (day['index_price'] / day['strike'] > 0.5)]
        day = day[day['tt'] > 0.1]
        day = day.groupby(['strike', 'time_expire']).mean()
        day = day.reset_index()
        # data = data[['strike', 'tt', 'index_price', 'irate', 'C_market']]

        weights = None

        # calibrate_Heston(day, t, weights)
        calibrate_Merton(day, t, weights)

    data = pd.read_csv('Data/data.csv')
    data.date = pd.to_datetime(data.date)


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
