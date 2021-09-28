import numpy
import pandas as pd
# import mlfinlab as ml
import os

class Label:

    def __init__(self, data):
        self.data = data

    def get_vol(self, delta, span = 100):
      # 1. compute returns of the form p[t]/p[t-1] - 1

      # 1.1 find the timestamps of p[t-1] values
      df0 = self.data.index.searchsorted(self.data.close.index - delta)
      df0 = df0[df0 > 0]

      # 1.2 align timestamps of p[t-1] to timestamps of p[t]
      # d = self.data.close.index[df0 - 1]
      df0 = pd.Series(self.data.index[df0-1], index=self.data.index[self.data.close.shape[0]-df0.shape[0]:])

      # 1.3 get values by timestamps, then compute returns
      df0 = self.data.close.loc[df0.index] / self.data.close.loc[df0.values].values - 1

      ## Some debugging variables
      # df0Index = numpy.array(df0.index.to_pydatetime(), dtype=numpy.datetime64)
      # df0Values = df0.values
      # a = self.data
      # b = self.data.loc[df0Values]
      # c = self.data.loc[df0Index]

      # 2. estimate rolling standard deviation
      df0 = df0.ewm(span = span).std()

      return df0

    def get_horizons(self, delta):
        t1 = self.data.index.searchsorted(self.data.close.index + delta)
        t1 = t1[t1 < self.data.close.shape[0]]
        t1 = self.data.index[t1]
        t1 = pd.Series(t1, index=self.data.index[:t1.shape[0]])
        return t1

    def get_touches(self, events, factors=[2, 1]):
      #
      # events: pd dataframe with columns
      #   t1: timestamp of the next horizon
      #   threshold: unit height of top and bottom barriers
      #   side: the side of each bet
      # factors: multipliers of the threshold to set the height of
      #          top/bottom barriers

      out = events[['t1']].copy(deep=True)
      if factors[0] > 0: thresh_uppr = factors[0] * events['threshold']
      else: thresh_uppr = pd.Series(index=events.index) # no uppr thresh
      if factors[1] > 0: thresh_lwr = -factors[1] * events['threshold']
      else: thresh_lwr = pd.Series(index=events.index)  # no lwr thresh
      for loc, t1 in events['t1'].iteritems():
        df0=self.data.close[loc:t1]                              # path prices
        df0=(df0 / self.data.close[loc] - 1) * events.side[loc]  # path returns
        out.loc[loc, 'stop_loss'] = \
          df0[df0 < thresh_lwr[loc]].index.min()  # earliest stop loss
        out.loc[loc, 'take_profit'] = \
          df0[df0 > thresh_uppr[loc]].index.min() # earliest take profit
      return out

    def get_labels(self, touches):
      out = touches.copy(deep=True)
      # pandas df.min() ignores NaN values
      first_touch = touches[['stop_loss', 'take_profit']].min(axis=1)
      for loc, t in first_touch.iteritems():
        if pd.isnull(t):
          out.loc[loc, 'label'] = 0
        elif t == touches.loc[loc, 'stop_loss']:
          out.loc[loc, 'label'] = -1
        else:
          out.loc[loc, 'label'] = 1
      return out


    def generate(self, window_size):
        delta = pd.Timedelta(minutes=window_size)
        self.data = self.data.assign(threshold=self.get_vol(delta)).dropna()
        self.data = self.data.assign(t1=self.get_horizons(delta)).dropna()
        events = self.data[['t1', 'threshold']]
        events = events.assign(side=pd.Series(1., events.index))  # long only
        touches = self.get_touches(events, [1, 1])
        touches = self.get_labels(touches)
        self.data = self.data.assign(label=touches.label)
        return self.data