from Metalabelling.TBM import getEvents, getBins

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import talib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

estim = 4000     # number of estimators for random forest
ver = 'V1-3'     # version number of the test for writing to file

if not os.path.exists('./Data'):
    os.mkdir('./Data/Variables')
    os.mkdir('./Data/Models')

## Import RawData and choose the desired features
raw_data = pd.read_csv('/Users/elefteriubogdan/PycharmProjects/TradingAlgo/data/raw/BTCUSDT_5m_raw_data.csv',
                       index_col=0, header=0)
raw_data.index = pd.to_datetime(raw_data.index)
data = raw_data[['close']].copy()

## Split the Dataset for the pipeline, compute features and save
if not os.path.isfile("./Data/Variables/features_201819.pk"):
    data_t1 = data[pd.to_datetime("2018-01-01 00:00:00"):pd.to_datetime("2019-01-01 00:00:00")]
    data_t1['RSI'] = talib.RSI(data_t1.close, timeperiod = 14)
    data_t1['ema20'] = talib.EMA(data_t1.close, timeperiod = 20)
    data_t1['ema50'] = talib.EMA(data_t1.close, timeperiod = 50)
    data_t1 = data_t1.dropna()
    data_t1.to_pickle("./Data/Variables/features_201819.pkl")
if not os.path.isfile("./Data/Variables/features_201920.pk"):
    data_t2 = data[pd.to_datetime("2019-01-01 00:00:00"):pd.to_datetime("2020-01-01 00:00:00")]
    data_t2['RSI'] = talib.RSI(data_t2.close, timeperiod = 14)
    data_t2['ema20'] = talib.EMA(data_t2.close, timeperiod = 20)
    data_t2['ema50'] = talib.EMA(data_t2.close, timeperiod = 50)
    data_t2 = data_t2.dropna()
    data_t2.to_pickle("./Data/Variables/features_201920.pkl")
if not os.path.isfile("./Data/Variables/features_202021.pk"):
    data_test = data[pd.to_datetime("2020-01-01 00:00:00"):pd.to_datetime("2021-01-01 00:00:00")]
    data_test['RSI'] = talib.RSI(data_test.close, timeperiod = 14)
    data_test['ema20'] = talib.EMA(data_test.close, timeperiod = 20)
    data_test['ema50'] = talib.EMA(data_test.close, timeperiod = 50)
    data_test = data_test.dropna()
    data_test.to_pickle("./Data/Variables/features_202021.pkl")

## Parameters for choosing the horizontal and vertical barriers
pt_sl = [1,1]
min_ret = 0.0005    ## 0.05%
delta_vol = pd.Timedelta(minutes=25)

## Compute horizontal barriers and calculate the sides
if not os.path.isfile("./Data/Variables/events_201819.pkl"):
    data_t1 = pd.read_pickle("./Data/Variables/features_201819.pkl")
    events_t1 = getEvents(data=data_t1.close,
                      ptSl=pt_sl,
                      delta=delta_vol,
                      minRet=min_ret,
                      t1=False,
                      side=None)
    events_t1.to_pickle("./Data/Variables/events_201819.pkl")
if not os.path.isfile("./Data/Variables/events_201920.pkl"):
    data_t2 = pd.read_pickle("./Data/Variables/features_201920.pkl")
    events_t2 = getEvents(data=data_t2.close,
                      ptSl=pt_sl,
                      delta=delta_vol,
                      minRet=min_ret,
                      t1=False,
                      side=None)
    events_t2.to_pickle("./Data/Variables/events_201920.pkl")
if not os.path.isfile("./Data/Variables/events_202021.pkl"):
    data_test = pd.read_pickle("./Data/Variables/features_202021.pkl")
    events_test = getEvents(data=data_test.close,
                      ptSl=pt_sl,
                      delta=delta_vol,
                      minRet=min_ret,
                      t1=False,
                      side=None)
    events_test.to_pickle("./Data/Variables/events_202021.pkl")


## PRIMARY MODEL (M1) - SIDE Prediction
# Train M1 with data_t1 and save
if not os.path.isfile("./Data/Models/rf_M1"+ver+"V1-1_201819.joblib"):
    data_t1 = pd.read_pickle('./Data/Variables/features_201819.pkl')
    events_t1 = pd.read_pickle('./Data/Variables/events_201819.pkl')
    events_t1['ret'], events_t1['side'] = getBins(data_t1.close, events_t1)

    X_t1 = data_t1.loc[events_t1.index].values
    y_t1 = events_t1.side.values.reshape(-1)

    clf = RandomForestClassifier(n_estimators=estim, max_features=4, max_depth=4)
    clf.fit(X_t1,y_t1)
    joblib.dump(clf, './Data/Models/rf_M1'+ver+'_201819.joblib')  # save the model
# Load M1 and make predictions on data_t2
if not os.path.isfile("./Data/Variables/pred_events"+ver+"_201920.pkl"):
    data_t2 = pd.read_pickle('./Data/Variables/features_201920.pkl')
    events_t2 = pd.read_pickle('./Data/Variables/events_201920.pkl')
    events_t2['ret'], events_t2['side'] = getBins(data_t2.close, events_t2)

    X_t2 = data_t2.loc[events_t2.index].values
    y_t2 = events_t2.side.values.reshape(-1)

    M1 = joblib.load('Data/Models/rf_M1'+ver+'_201819.joblib')
    y_t2_pred = M1.predict(X_t2)
    events_t2['pred_side'] = y_t2_pred

    # Metalabeling based on predictions of SIDE on data_t2
    events_t2['realised_ret'], events_t2['trades'] = getBins(data_t2.close, events_t2)
    events_t2.to_pickle('./Data/Variables/pred_events'+ver+'_201920.pkl')

## SECONDARY MODEL (M2) - SIZE Prediction
# Train M2 with data_t2+pred_side and save
if not os.path.isfile("Data/Models/rf_M2"+ver+"_201920.joblib"):
    events_t2 = pd.read_pickle('Data/Variables/pred_events'+ver+'_201920.pkl')
    data_t2 = pd.read_pickle('./Data/Variables/features_201920.pkl')
    X_t2_M2 = np.hstack((data_t2.loc[events_t2.index].values,
                         np.expand_dims(events_t2.pred_side, axis=1)))
    y_t2_M2 = events_t2.trades.values.reshape(-1)

    M2 = RandomForestClassifier(n_estimators=estim, max_features=5, max_depth=5)
    M2.fit(X_t2_M2, y_t2_M2)
    joblib.dump(M2, 'Data/Models/rf_M2'+ver+'_201920.joblib')  # save the model



## TESTING - Load M1&M2 and data_test for a complete pipeline test
if not os.path.isfile("Data/Variables/pred_events"+ver+"_202021.pkl"):
    M1 = joblib.load('Data/Models/rf_M1'+ver+'_201819.joblib')
    M2 = joblib.load('Data/Models/rf_M2'+ver+'_201920.joblib')

    data_test = pd.read_pickle('./Data/Variables/features_202021.pkl')
    events_test = pd.read_pickle('./Data/Variables/events_202021.pkl')

    events_test['ret'], events_test['side'] = getBins(data_test.close, events_test)
    X_testM1 = data_test.loc[events_test.index].values

    y_test_predM1 = M1.predict(X_testM1)
    events_test['pred_side'] = y_test_predM1
    events_test['realised_ret'], events_test['trades'] = getBins(data_test.close, events_test)

    X_testM2 = np.hstack((X_testM1, np.expand_dims(events_test.pred_side, axis=1)))
    y_test_predM2 = M2.predict(X_testM2)
    events_test['pred_trades'] = y_test_predM2
    events_test.to_pickle('./Data/Variables/pred_events'+ver+'_202021.pkl')

# Test the prediction of SIDE and plot confusion matrix
events = pd.read_pickle('Data/Variables/pred_events'+ver+'_202021.pkl')
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














