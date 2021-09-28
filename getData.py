import pandas as pd
import math
import os.path
from binance.client import Client
from datetime import timedelta, datetime, date

binance_api_key = 'c4aqjt0eGrzPFPh8QgggNZ3UvNz0Xq0eG4UE5Y6Cy8V89tcw5cPAmrgxaa41Cl57'
binance_api_secret = 'CwTBAKDnC4I3MyCq0elzxxNrR21IbaRYEtiDVrIW74skGcQjLNwqjdEbaxVkSsow'

kline_sizes = {"1m": 1, "5m": 5, '30m': 30, "1h": 60, "1d": 1440}
batch_size = 750
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

def download_binance_data(symbol, kline_size, start_date='1 Jan 2017'):
    """
        Function that downloads raw data from binance API and saves to file. If file already exists, fetches
        and appends new data.
        - symbol: Ticker Symbol
        - kline_size: Size of Klines
        - start_date: optional start date
       :return:
    """
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./data/raw'):
        os.mkdir('./data/raw')

    filename = './data/raw/%s-%s-raw-data.csv' % (symbol, kline_size)
    data = pd.read_csv(filename, index_col=0, header=0) if os.path.isfile(filename) else pd.DataFrame()
    start_time = datetime.strptime(start_date, '%d %b %Y')
    today = date.today()
    end_time = datetime(today.year, today.month, today.day)

    if len(data) > 0:
        start_time = datetime.strptime(data.iloc[-1].name, '%Y-%m-%d %H:%M:%S') + timedelta(minutes=kline_sizes[kline_size])

    if start_time < end_time:
        delta_min = (end_time - start_time).total_seconds()/60
        available_data = math.ceil(delta_min/kline_sizes[kline_size])
        print('Downloading %d instances of %s data since %s' % (available_data, kline_size, start_time))

        klines = binance_client.get_historical_klines(symbol, kline_size, start_time.strftime("%d %b %Y %H:%M:%S"), end_time.strftime("%d %b %Y %H:%M:%S"))
        data_new = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
        data_new['timestamp'] = pd.to_datetime(data_new['timestamp'], unit='ms')
        data_new.set_index('timestamp', inplace=True)
        data = data.append(data_new, sort=True)
        data.to_csv(filename)

    print('All caught up..!')
    return
