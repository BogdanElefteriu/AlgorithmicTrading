import datetime
import ta
import matplotlib
import os
import sys
import pandas as pd
import tables
import numpy as np
from getData import download_binance_data
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import normalize
from distMat import ImageBuilder
import time
from labelling import Label


def preprocess(raw_data_path='./data/raw/BTCUSDT_5m_raw_data.csv'):
    """
    Function that reads in raw data from csv, calculates indicators and generates distance matrices and training labels.
    :param raw_data_path: path of raw data file
    :return: Saved data in h5 file
    """
    if not os.path.isfile(raw_data_path):
        print("Raw data file does not exist")
        sys.exit()
    if not os.path.exists('./data/preprocessed'):
        os.mkdir('./data/preprocessed')

    start_time = time.time()
    strings = raw_data_path.split('/')[-1].split('_')
    pair = strings[0]
    time_interval = strings[1]

    raw_data = pd.read_csv(raw_data_path, index_col=0, header=0)

    # Pick the data from the dataset
    data = raw_data[['open', 'high', 'low', 'close', 'volume']].copy()

    # Calculate Indicators
    indicators = pd.DataFrame()
    indicators['%K'] = ta.momentum.stoch(data.high, data.low, data.close, window=14, smooth_window=3)
    indicators['%D'] = indicators['%K'].rolling(3).mean()
    indicators['RSI'] = ta.momentum.rsi(data.close, window=14)
    indicators['MACD'] = ta.trend.macd_diff(data.close)
    indicators['MACDS'] = ta.trend.macd_signal(data.close)

    data = pd.concat([data, indicators], axis=1)
    data.dropna(inplace=True)

    # Generate distance matrices
    img_size = 200 # in 5 minute intervals
    total_time = data.close.size
    sigma = 2  # smoothening factor

    data_file = tables.open_file('./data/preprocessed/{}_{}_data.h5'.format(pair, time_interval), "w")
    filters = tables.Filters(complevel=5, complib='blosc')
    data_file.create_group("/", 'data', 'Distance Matrix Images')

    for signal in ['close', 'volume', 'RSI', 'MACD', 'MACDS']:
        data_storage = data_file.create_earray(data_file.root.data, signal, tables.Float32Atom(),
                                        shape=(0, img_size, img_size), filters=filters, expectedrows=total_time - img_size)
        for i in range(0, total_time - img_size + 1):
            dist_mat = euclidean_distances(data[signal].iloc[i:img_size+i].values.reshape(-1, 1))
            image = gaussian_filter(dist_mat, sigma=sigma)
            image = normalize(image, norm='max')
            data_storage.append(np.expand_dims(image, 0).astype('float32'))
            if i % 10000 == 0:
                print("{} {}/{}".format(signal, i+1, total_time-img_size+1))

    data_file.close()

    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    return

