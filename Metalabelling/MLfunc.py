import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

from Metalabelling.TBM import getBins, getEvents

import pandas as pd
import numpy as np
import joblib
import os


def folderStruct(ver):
    if not os.path.exists('./Data'):
        os.mkdir('./Data')
        os.mkdir('Data/Raw')
        os.mkdir('Data/Tests')
    if not os.path.exists('./Data/Tests/'+ver):
        os.mkdir('./Data/Tests/'+ver)

def computeSide(ver, years, pt_sl, delta_vol, min_ret):
    data = pd.read_pickle("./Data/Raw/features_"+years+".pkl")
    events = getEvents(data=data.close, ptSl=pt_sl, delta=delta_vol, minRet=min_ret, trigger=False, side=None)
    events.to_pickle("./Data/Tests/"+ver+"/events_"+years+".pkl")

def NN(features):
    model = Sequential()
    model.add(Dense(256, input_dim=features, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def trainM1(ver, years, estim, features = 4):
    data = pd.read_pickle('./Data/Raw/features_'+years+'.pkl')
    event = pd.read_pickle('./Data/Tests/'+ver+'/events_'+years+'.pkl')
    event['ret'], event['side'] = getBins(data.close, event)

    X = data.loc[event.index].values
    #RF
    y = event.side.values.reshape(-1)
    # #NN
    # y = tf.one_hot(event.side.values.reshape(-1), depth=2)

    ## Random Forest
    M1 = RandomForestClassifier(n_estimators=estim, max_features=features, max_depth=features)
    M1.fit(X, y)
    joblib.dump(M1, './Data/Tests/'+ver+'/rf_M1_'+years+'.joblib')  # save the model

    # ## Neural Network
    # M1 = NN(features)
    # M1.fit(X, y, epochs=40, batch_size=500)
    # M1.save('./Data/Tests/'+ver+'/nn_M1_'+years+'.h5')

def predictM1(ver, years):
    data = pd.read_pickle('./Data/Raw/features_'+years+'.pkl')
    event = pd.read_pickle('./Data/Tests/'+ver+'/events_'+years+'.pkl')
    event['ret'], event['side'] = getBins(data.close, event)

    X = data.loc[event.index].values
    y = event.side.values.reshape(-1)

    m1 = joblib.load('./Data/Tests/'+ver+'/rf_M1_'+years+'.joblib')
    y_pred = m1.predict(X)
    event['pred_side'] = y_pred

    # Metalabeling based on predictions of SIDE on data_t2
    event['realised_ret'], event['trades'] = getBins(data.close, event)
    event.to_pickle('./Data/Tests/'+ver+'/pred_events_'+years+'.pkl')

def trainM2(ver, years, estim = 100, features = 5):
    data = pd.read_pickle('./Data/Raw/features_'+years+'.pkl')
    event = pd.read_pickle('Data/Tests/'+ver+'/pred_events_'+years+'.pkl')

    # data+side as input vector
    X_M2 = np.hstack((data.loc[event.index].values,
                    np.expand_dims(event.pred_side, axis=1)))
    # meta-labels as label vector-RF
    y_M2 = event.trades.values.reshape(-1)
    # # meta-labels as label vector-NN
    # y_M2 = tf.one_hot(event.trades.values.reshape(-1), depth=2)

    # Random Forest
    M2 = RandomForestClassifier(n_estimators=estim, max_features=features, max_depth=features)
    M2.fit(X_M2, y_M2)
    joblib.dump(M2, 'Data/Tests/' + ver + '/rf_M2_' + years + '.joblib')  # save the model

    # ## Neural Network
    # M2 = NN(features)
    # M2.fit(X_M2, y_M2, epochs=10, batch_size=100)
    # M2.save('./Data/Tests/'+ver+'/nn_M2_'+years+'.h5')

def predictPipeline(ver, years, testing_years):
    M1 = joblib.load('./Data/Tests/'+ver+'/rf_M1_'+years+'.joblib')
    M2 = joblib.load('./Data/Tests/'+ver+'/rf_M2_'+years+'.joblib')

    data = pd.read_pickle('./Data/Raw/features_'+testing_years+'.pkl')
    event = pd.read_pickle('./Data/Tests/'+ver+'/events_'+testing_years+'.pkl')
    event['ret'], event['side'] = getBins(data.close, event)

    X_M1 = data.loc[event.index].values
    y_M1 = event.side.values.reshape(-1)

    side_pred = M1.predict(X_M1)
    event['pred_side'] = side_pred
    event['realised_ret'], event['trades'] = getBins(data.close, event)

    X_M2 = np.hstack((X_M1, np.expand_dims(event.pred_side, axis=1)))
    size_pred = M2.predict(X_M2)
    event['pred_trades'] = size_pred
    event.to_pickle('./Data/Tests/'+ver+'/pred_events_'+testing_years+'.pkl')

def plotLabels(data, events, datapoints = 1000, predicted = False):
    # If predicted is False, the function will plot the calculated side and the associated metalabels (Groundtruths)
    # If predicted is True, the function will plot the predicted side and predicted metalabels (Predictions)

    ## Plot METALABEL & SIDE on price for model 1 prediction on data_m2
    fig, ax = plt.subplots()
    ax.plot(data.close[:datapoints])
    if predicted:
        trades = events.pred_trades
        side = events.pred_side
    else:
        trades = events.trades
        side = events.side

    for i, label in enumerate(trades[:datapoints]):
        if label != 0:
            if side[i] > 0:
                ax.annotate(label, xy=(data.index[i], data.close.loc[data.index[i]]),
                            color='green')
            else:
                ax.annotate(label, xy=(data.index[i], data.close.loc[data.index[i]]),
                            color='red')
    return plt.show()
