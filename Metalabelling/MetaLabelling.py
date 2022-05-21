from Metalabelling.TBM import getEvents, getBins
from Metalabelling.MLfunc import folderStruct, computeSide, \
    trainM1, trainM2, predictM1, predictPipeline, plotLabels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import talib
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

ver = 'v1-6'     # version number of the test for writing to file
years = '2018-19'   # the dataset used {2018-19, 2019-20, 2020-21}
testing_years = '2020-21'

## Define the Data folder structure
folderStruct(ver)

## Import RawData and choose the desired features
raw_data = pd.read_csv('/Users/elefteriubogdan/PycharmProjects/TradingAlgo/data/raw/BTCUSDT_5m_raw_data.csv',
                       index_col=0, header=0)
raw_data.index = pd.to_datetime(raw_data.index)
data = raw_data[['close']].copy()

## Split the Dataset for the pipeline, compute features and save
if not os.path.isfile("./Data/Raw/features_2018-19.pkl"):
    data_t1 = data[pd.to_datetime("2018-01-01 00:00:00"):pd.to_datetime("2019-01-01 00:00:00")]
    data_t1['RSI'] = talib.RSI(data_t1.close, timeperiod = 14)
    data_t1['ema20'] = talib.EMA(data_t1.close, timeperiod = 20)
    data_t1['ema50'] = talib.EMA(data_t1.close, timeperiod = 50)
    data_t1 = data_t1.dropna()
    data_t1.to_pickle("./Data/Raw/features_2018-19.pkl")
if not os.path.isfile("./Data/Raw/features_2019-20.pkl"):
    data_t2 = data[pd.to_datetime("2019-01-01 00:00:00"):pd.to_datetime("2020-01-01 00:00:00")]
    data_t2['RSI'] = talib.RSI(data_t2.close, timeperiod = 14)
    data_t2['ema20'] = talib.EMA(data_t2.close, timeperiod = 20)
    data_t2['ema50'] = talib.EMA(data_t2.close, timeperiod = 50)
    data_t2 = data_t2.dropna()
    data_t2.to_pickle("./Data/Raw/features_2019-20.pkl")
if not os.path.isfile("./Data/Raw/features_2020-21.pkl"):
    data_test = data[pd.to_datetime("2020-01-01 00:00:00"):pd.to_datetime("2021-01-01 00:00:00")]
    data_test['RSI'] = talib.RSI(data_test.close, timeperiod = 14)
    data_test['ema20'] = talib.EMA(data_test.close, timeperiod = 20)
    data_test['ema50'] = talib.EMA(data_test.close, timeperiod = 50)
    data_test = data_test.dropna()
    data_test.to_pickle("./Data/Raw/features_2020-21.pkl")

    ## Parameters for choosing the horizontal and vertical barriers
pt_sl = [1,1]
min_ret = 0.0005    ## 0.05%
delta_vol = pd.Timedelta(minutes=25)

## Compute horizontal barriers and calculate the sides
if not os.path.isfile("./Data/Tests/"+ver+"/events_"+years+".pkl"):
    computeSide(ver=ver, years=years, pt_sl=pt_sl, delta_vol=delta_vol, min_ret=min_ret)

if not os.path.isfile("./Data/Tests/"+ver+"/events_"+testing_years+".pkl"):
    computeSide(ver=ver, years=testing_years, pt_sl=pt_sl, delta_vol=delta_vol, min_ret=min_ret)

## PRIMARY MODEL (M1) - SIDE Prediction
estim = 300     # number of estimators for random forest

# Train M1 and save
if not os.path.isfile("./Data/Tests/"+ver+"/rf_M1_"+years+".joblib"):
    trainM1(ver=ver, years=years, estim=estim, features=4)

# Load M1 and make predictions -> Meta-label the predictions {0, 1}
if not os.path.isfile("./Data/Tests/"+ver+"/pred_events_"+years+".pkl"):
    predictM1(ver=ver, years=years)


## SECONDARY MODEL (M2) - SIZE Prediction
# Train M2 with [data+pred_side & meta-labels] and save
if not os.path.isfile("./Data/Tests/"+ver+"/rf_M2_"+years+".joblib"):
    trainM2(ver=ver, years=years, estim=estim, features=5)


## TESTING - Load M1&M2 and data_test for a complete pipeline test
if not os.path.isfile("./Data/Tests/"+ver+"/pred_events_"+testing_years+".pkl"):
    predictPipeline(ver=ver, years=years, testing_years=testing_years)

# Test the prediction of SIDE and plot confusion matrix
events_test = pd.read_pickle('./Data/Tests/'+ver+'/pred_events_'+testing_years+'.pkl')
data_test = pd.read_pickle('./Data/Raw/features_'+testing_years+'.pkl')

plotLabels(data_test,events_test,2000,predicted=False)
plotLabels(data_test,events_test,2000,predicted=True)

conf_matrix1 = confusion_matrix(y_true=events.side, y_pred=events.pred_side)
conf_matrix2 = confusion_matrix(y_true=events.trades, y_pred=events.pred_trades)

prec1 = precision_score(y_true=events.side, y_pred=events.pred_side)
rec1 = recall_score(y_true=events.side, y_pred=events.pred_side)
f1 = f1_score(y_true=events.side, y_pred=events.pred_side)

prec2 = precision_score(y_true=events.trades, y_pred=events.pred_trades)
rec2 = recall_score(y_true=events.trades, y_pred=events.pred_trades)
f2 = f1_score(y_true=events.trades, y_pred=events.pred_trades)

print ('Precision of M1: {}\nRecall of M1: {}'.format(prec1, rec1))
print ('F1-Score of M1: {}'.format(f1))
print ('Precision of M2: {}\nRecall of M2: {}'.format(prec2, rec2))
print ('F1-Score of M2: {}'.format(f2))

fig, ax = plt.subplots(1,2)
ax[0].matshow(conf_matrix1, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix1.shape[0]):
    for j in range(conf_matrix1.shape[1]):
        ax[0].text(x=j, y=i, s=conf_matrix1[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
ax[0].set_title('Confusion Matrix Side', fontsize=18)

ax[1].matshow(conf_matrix2, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix2.shape[0]):
    for j in range(conf_matrix2.shape[1]):
        ax[1].text(x=j, y=i, s=conf_matrix2[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
ax[1].set_title('Confusion Matrix Size', fontsize=18)
plt.show()
