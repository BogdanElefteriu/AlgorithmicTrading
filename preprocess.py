import time
import ta
import os
import math
import pandas as pd
import tables
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import normalize

from getData import download_binance_data
from labelling import Label

## PARAMETERS
start_time = time.time()

# Labelling
window_size = 500

# Distance Matrix
img_size = 200 # in 5 minute intervals
sigma = 2  # smoothening factor

# Shorten data for testing
start = pd.to_datetime("2021-09-01 00:00:00")
end = pd.to_datetime("2021-09-05 00:00:00")


def preprocess(raw_data_path= './data/raw/BTCUSDT_5m_raw_data.csv'):
    """
    Function that reads in raw data from csv, calculates indicators and generates distance matrices and training labels.
    :param raw_data_path: path of raw data file
    :return: Saved data in h5 file
    """

    if not os.path.isfile(raw_data_path):
        print("Raw data file does not exist")
        print('Downloading and saving raw data file')
        download_binance_data(symbol= 'BTCUSDT', kline_size='5m', start_date= '1 Jan 2017')

    if not os.path.exists('./data/preprocessed'):
        os.mkdir('./data/preprocessed')

    strings = raw_data_path.split('/')[-1].split('_')
    pair = strings[0]
    time_interval = strings[1]

    ### Import raw data
    raw_data = pd.read_csv(raw_data_path, index_col=0, header=0)

    # Transform index timestamps into DateTimeIndex objects for better handling
    raw_data.index = pd.to_datetime(raw_data.index)

    # Pick the data from the dataset
    data = raw_data[['open', 'high', 'low', 'close', 'volume']].copy()

    # Eliminates any duplicate datapoints in the dataset
    for i in data.loc[data.index.duplicated(), :].iterrows():
        if data.loc[i[0]] is not None:
            print(data.loc[i[0]])
        data = data.drop(labels=data.loc[i[0]].iloc[1:].index, axis=0)

    # Shorten for testing
    if start and end is not None:
        data = data.loc[start:end]


    ### Generate Indicators
    indicators = pd.DataFrame()
    indicators['%K'] = ta.momentum.stoch(data.high, data.low, data.close, window=14, smooth_window=3)
    indicators['%D'] = indicators['%K'].rolling(3).mean()
    indicators['RSI'] = ta.momentum.rsi(data.close, window=14)
    indicators['MACD'] = ta.trend.macd_diff(data.close)
    indicators['MACDS'] = ta.trend.macd_signal(data.close)

    data = pd.concat([data, indicators], axis=1)


    ### Generate labels for each image in the dataset and arrange each label with its associated image
    label = Label(data.close)
    labels = label.generate(window_size=window_size)
    data = pd.merge_asof(left=data, right=labels[['label']], right_index=True, left_index=True,
                         direction='nearest', tolerance=pd.Timedelta('1 second'))
    data.dropna(inplace=True)

    # Arrange labels with the images
    data.label = data.label[img_size-1:]

    # Open the datafile in preparation for writing label and image data to the disk
    data_file = tables.open_file('./data/preprocessed/{}_{}_data.h5'.format(pair, time_interval), mode="w")
    filters = tables.Filters(complevel=5, complib='blosc')
    data_file.create_group("/", 'data', 'Distance Matrix Images')

    # Save the labels on the disk
    label_storage = data_file.create_carray(data_file.root, 'label', tables.Int8Atom(),
                                            shape = (1,data.label.shape[0]-img_size+1),  filters = filters)

    labels = data.label
    labels.dropna(inplace=True)
    label_storage[:] = np.array(labels)

    ### Generate distance matrices
    total_time = data.close.size

    # Save the images on the disk
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
    return data
