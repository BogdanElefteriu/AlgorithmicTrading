from getData import Get_binance
from distMat import ImageBuilder
from labelling import Label

import pandas as pd
import numpy as np
import os
import datetime
import ta
import matplotlib
import sys

###################################### DATA MANIPULATION ######################################

        ## Set the currency and timeframe to work with
pair = 'BTCUSDT'
time = '5m'

## Download and update the dataset from Binance API
# all_data = Get_binance(pair, time, save = True)         ## Uncomment just to download new data or for the first time running the script

## Open downloaded data and arrange timestamp
raw_data = pd.read_csv(os.getcwd()+'/data/raw/' + pair + '-' + time + '-data.csv')
raw_data.set_index('timestamp', inplace=True)
raw_data.index = pd.to_datetime(raw_data.index)

## Pick the data from the dataset
data = pd.DataFrame()
data = raw_data[['open', 'high', 'low', 'close', 'volume']].copy()

## Shorten for testing
start = pd.to_datetime(str('2020-08-01'))
end = pd.to_datetime(str('2020-09-04'))
data = data.loc[start:end]

## Eliminates any duplicate datapoints in the dataset
for i in data.loc[data.index.duplicated(),:].iterrows():
    # print(data.loc[i[0]])
    data = data.drop(labels = data.loc[i[0]].iloc[1:].index, axis = 0)



# ###################################### TECHNICAL INDICATORS ######################################
#
# indicators = pd.DataFrame()
# indicators['%K'] = ta.momentum.stoch(data.high, data.low, data.close, window=14, smooth_window=3)
# indicators['%D'] = indicators['%K'].rolling(3).mean()
# indicators['RSI'] = ta.momentum.rsi(data.close, window=14)
# indicators['MACD'] = ta.trend.macd_diff(data.close)
# indicators['MACDS'] = ta.trend.macd_signal(data.close)
#
# data = pd.concat([data, indicators[['%K']],
#                         indicators[['%D']],
#                         indicators[['RSI']],
#                         indicators[['MACD']],
#                         indicators[['MACDS']]], axis = 1)
# data.dropna(inplace=True)
#
#
#
# ###################################### IMAGE CREATION ######################################
#
# # Generate distance matrices
# imageSize = 200             # in 5 minute intervals
# totalTime = data.close.size
# sigma = 2                  # smoothening factor
#
# closeImage = ImageBuilder(data.close)
# closeImages = closeImage.dist_mat(imageSize, totalTime, sigma)
# normcloseImages = closeImage.normalise(closeImages, imageSize, totalTime)
#
# volImage = ImageBuilder(data.volume)
# volImages = volImage.dist_mat(imageSize, totalTime, sigma)
# normvolImages = volImage.normalise(volImages, imageSize, totalTime)
#
# RSIImage = ImageBuilder(data.RSI)
# RSIImages = RSIImage.dist_mat(imageSize, totalTime, sigma)
# normRSIImages = RSIImage.normalise(RSIImages, imageSize, totalTime)
#
# MACDImage = ImageBuilder(data.MACD)
# MACDImages = MACDImage.dist_mat(imageSize, totalTime, sigma)
# normMACDImages = MACDImage.normalise(MACDImages, imageSize, totalTime)
#
# MACDSImage = ImageBuilder(data.MACDS)
# MACDSImages = MACDSImage.dist_mat(imageSize, totalTime, sigma)
# normMACDSImages = MACDSImage.normalise(MACDSImages, imageSize, totalTime)
#
# normcombinedMACD = (normMACDSImages) / (normMACDImages)
#
# allImages = np.stack((normcloseImages, normvolImages, normcombinedMACD, normRSIImages), axis = -1)
#
#
# matplotlib.image.imsave('Test1Photo.jpg', normcloseImages[45,:,:])