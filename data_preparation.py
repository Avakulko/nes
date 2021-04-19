import pandas as pd
import numpy as np

# timestamp: ms passed from epoch
# price: option on 1 BTC price in BTC
# instrument_name: BTC-17FEB17-975-C
# index_price: price of 1 BTC in USD
# direction: sell/buy
# amount: amount of underlying BTC
# time_trade: ...
# strike: strike price in USD
# type: call/put
# time_create: ...
# time_expire: ...
# date: date of trade
# irate: 3m Treasury bill rate
# price_USD: option on 1 BTC price in USD
# iv_Tbill: BS implied volatility according to 3m Treasury Bill
# tt: time to maturity in fractions of a year (365 day)


# C_market: рыночная стоимость колл-опциона на 1 BTC выраженная в USD

df = pd.read_csv('Data/Deribit_w_IV.csv')

# некоторые iv_tbill имеют значение .
df = df[df['iv_Tbill'] != '.']
df['iv_Tbill'] = df['iv_Tbill'].astype(np.float64)
# Выбрасываем безумные значения IV. У всех таких ордеров очень маленькое tt. Откуда такие значения IV и нужно ли их выбрасывать?
df = df[df['iv_Tbill'] <= 300]
# Корректируем типы
df['date'] = pd.to_datetime(df['date'])
df['time_trade'] = pd.to_datetime(df['time_trade'])
df['time_create'] = pd.to_datetime(df['time_create'])
df['time_expire'] = pd.to_datetime(df['time_expire'])

# Переводим стоимости путов в стоимости коллов через пут-колл паритет и помещаем результат в C_market
puts = df[df['type'] == 'put']
calls = df[df['type'] == 'call']
calls['C_market'] = calls['price_USD']
puts['C_market'] = puts['price_USD'] + puts['index_price'] - puts['strike'] * np.exp(-puts['irate'] * puts['tt'])
df = pd.concat([calls, puts])
df = df.sort_values(by='date')
df = df.reset_index()

df.to_csv('Data/data.csv', index=False)