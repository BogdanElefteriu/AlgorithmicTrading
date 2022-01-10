import pandas as pd
import numpy as np

def get_volatility(data, delta, span=25):
    """
    Function that computes the volatility in the price based on the returns: (p[t]/p[t-1]) - 1 in a specified timeframe.
    Volatility is computed by taking the rolling standard deviation of the returns.
    This function helps to define the threshold while defining BUY/SELL labels (i.e. the width of the channel).
    :param data: close price of the asset
    :param delta: window size in minutes
    :param span: the decay in terms of span for the exponential weighted average
    :return: threshold value
    """
    # Find the timestamps of p[t-1] values
    timestamp = data.index.searchsorted(pd.to_datetime(data.index) - delta)
    timestamp = timestamp[timestamp > 0]

    # Align timestamps of p[t-1] to timestamps of p[t]
    timestamp = pd.Series(data.index[timestamp - 1],
                          index=data.index[data.shape[0] - timestamp.shape[0]:])

    # Get values by timestamps, then compute returns
    returns = (data.loc[timestamp.index] / data.loc[timestamp.values].values) - 1

    # Estimate rolling standard deviation
    threshold = returns.ewm(span=span).std()
    threshold = threshold.dropna()
    return threshold

def get_horizons(data, delta):
    """
    Function that computes the timestamps at the end of each window size.
    :param delta: window size in minutes
    :return: array of timestamps defining the end of window sizes
    """
    # Find the timestamps of the end of window size
    horizon = data.index.searchsorted(data.index + delta)
    horizon = horizon[horizon < data.shape[0]]

    # Align the start-end timestamps of the window
    horizon = pd.Series(data.index[horizon], index=data.index[:horizon.shape[0]])
    return horizon

def applyPtSl(data, events, ptSl):
    """
    Function that computes the vertical barrier threshold levels for each point in the
    dataset and then calculates the first timestamps at which either the upper or the
    lower barriers were hit first.
    :param data: close price of the asset
    :param events: DataFrame containing t1, volatility and side (if provided)
    :param ptSl:1x2 matrix containing the constants to define the threshold of the
                barriers based on a certain amount of std deviations of the volatility
    :return: out variable containing the timestamps at which each barrier was hit first
    """

    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    out = events[['t1']].copy(deep=True)

    # If the upper barrier is active(ptSl[0]>0) then compute its threshold based on volatility
    if ptSl[0]>0:
        pt = ptSl[0] * events['trgt']
    else:
        pt = pd.Series(index=events.index) # NaNs

    # If the lower barrier is active(ptSl[1]>0) then compute its threshold based on volatility
    if ptSl[1]>0:
        sl = -ptSl[1] * events['trgt']
    else:
        sl = pd.Series(index=events.index) # NaNs

    # Compute the timestamp at which the first (lower or upper) barrier was hit based on threshold
    for loc, t1 in events['t1'].fillna(data.index[-1]).iteritems():
        df = data[loc:t1] # path prices
        df = (df/data[loc]-1) * events.at[loc, 'side'] # path returns

        out.loc[loc, 'sl'] = df[df < sl[loc]].index.min() # earliest stop loss
        out.loc[loc, 'pt'] = df[df > pt[loc]].index.min() # earliest profit taking

    return out

def getEvents(data, ptSl, delta, minRet=0.005, t1=False, side=None):
    """
    This function defines the barriers threshold(trgt) as the volatility and uses it
    to calculate the timestamps at which the upper/lower barriers are hit.
    :param data: close price of the asset
    :param tEvents: trigger events for defining the start of the window
    :param ptSl: 1x2 matrix containing the constants to define the threshold of the
                barriers based on a certain amount of std deviations of the volatility
    :param delta: window size in minutes for volatility computation
    :param minRet: minimum return for a trade (0.05%
    :param t1: DataFrame containing the end timestamps of the window (if none, there is no vertical barrier)
    :param side: long/short(if none, the function is used for finding the side)
    :return: events DataFrame containing the timestamps of the profitable trade
    """
    # Compute volatility as a target for barriers
    trgt = get_volatility(data, delta)

    # Take out the data points where volatility is lower than minRet
    trgt = trgt[trgt > minRet]

    # If there is no window end (vertical barrier), then fill t1 with NaNs
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=trgt.index)

    # If there is no side provided, then fill side_ with 1's
    if side is None:
        side_ = pd.Series(1.,index=trgt.index)
        ptSl_ = [ptSl[0], ptSl[0]]
    else:
        side_ = side.loc[trgt.index]
        ptSl_ = ptSl[:2]

    events = (pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
              .dropna(subset=['trgt']))

    # Calculate the timestamps at which the barriers are hit given the volatility threshold
    df = applyPtSl(data, events, ptSl_)

    events['t1']=df.dropna(how='all').min(axis=1) #pd.min ignores nan

    # If there was no side given as an input, eliminate the
    # generated side(used only for calculation consistency) from the output
    if side is None:
        events = events.drop('side', axis=1)

    events = events.dropna()
    return events

def getBins(data, events):
    '''
    Compute event's outcome (including side information).
    Bin in (0,1) <-label by pnl (meta-labeling)
    data is the close price of the asset
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    # Align prices with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = data.reindex(px, method='bfill')

    # Create output object and calculate the path returns
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

    # Meta-labeling
    if 'pred_side' in events_:
        out['ret'] *= events_['pred_side']

    out['bin'] = np.sign(out['ret'])

    if 'pred_side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0

    return out['ret'], out['bin']