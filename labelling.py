import pandas as pd


class Label:

    def __init__(self, data):
        self.data = data

    def get_volatility(self, delta, span = 100):
        """
        Function that computes the volatility in the price based on the returns: (p[t]/p[t-1]) - 1 in a specified timeframe.
        Volatility is computed by taking the rolling standard deviation of the returns.
        This function helps to define the threshold while defining BUY/SELL labels (i.e. the width of the channel).
        :param delta: window size in minutes
        :param span: the decay in terms of span for the exponential weighted average
        :return: threshold value
        """
        # Find the timestamps of p[t-1] values
        timestamp = self.data.index.searchsorted(pd.to_datetime(self.data.index) - delta)
        timestamp = timestamp[timestamp > 0]

        # Align timestamps of p[t-1] to timestamps of p[t]
        timestamp = pd.Series(self.data.index[timestamp - 1],
                              index=self.data.index[self.data.shape[0] - timestamp.shape[0]:])

        # Get values by timestamps, then compute returns
        returns = (self.data.loc[timestamp.index] / self.data.loc[timestamp.values].values) - 1

        # Estimate rolling standard deviation
        threshold = returns.ewm(span = span).std()
        return threshold


    def get_horizons(self, delta):
        """
        Function that computes the timestamps at the end of each window size.
        :param delta: window size in minutes
        :return: array of timestamps defining the end of window sizes
        """
        horizon = self.data.index.searchsorted(self.data.index + delta)
        horizon = horizon[horizon < self.data.shape[0]]
        horizon = self.data.index[horizon]
        horizon = pd.Series(horizon, index=self.data.index[:horizon.shape[0]])
        return horizon


    def get_labels(self, variables, factors=[2, 1]):
        """
        Function that defines the thresholds based on multipliers(factors) and defines the timestamps where the price
        return has crossed the threshold levels. It then assigns the corresponding labels to the timestamps.
        :param variables: threshold values and horizon timestamps
        :param factors: multipliers of the threshold to set the height of the top/bottom barriers
        side: the side of each bet
        :return: variable containing labels and trigger timestamps for the respective labels
        """
        labels = variables[['horizon']].copy(deep=True)

        thresh_uppr = factors[0] * variables['threshold']       # Define the upper threshold values
        thresh_lwr = -factors[1] * variables['threshold']       # Define the lower threshold values

        for index, horizon in variables['horizon'].iteritems():
          returns = self.data[index:horizon]                                  # Define the path prices
          returns = (returns / self.data[index] - 1) * variables.side[index]  # Calculate price returns:(p[t]/p[t-1])-1

          labels.loc[index, 'sell'] = returns[returns < thresh_lwr[index]].index.min()  # Sell signals timestamps
          labels.loc[index, 'buy'] = returns[returns > thresh_uppr[index]].index.min()  # Buy signals timestamps

        # Assign the labels to their specific timestamp
        for index, timestamp in labels[['sell', 'buy']].min(axis=1).iteritems():       # df.min() ignores NaN values
            if pd.isnull(timestamp):
                labels.loc[index, 'label'] = 0

            elif timestamp == labels.loc[index, 'sell']:
                labels.loc[index, 'label'] = -1

            elif timestamp == labels.loc[index, 'buy']:
                labels.loc[index, 'label'] = 1
        return labels


    def generate(self, window_size):
        delta = pd.Timedelta(minutes=window_size)       # window size

        # Compute variables: horizon = the price at the end of the window size
        #                    threshold = threshold for defining a label (i.e. size of the vertical channel)
        #                    side = buy/sell only strategy OR hybrid strategy(--- to be implemented ---)
        variables = pd.DataFrame()
        variables = variables.assign(horizon=self.get_horizons(delta)).dropna()
        variables = variables.assign(threshold=self.get_volatility(delta)).dropna()
        variables = variables.assign(side=pd.Series(1., variables.index))  # long only

        labels = self.get_labels(variables, [1, 1])
        variables = variables.assign(label=labels.label)
        return variables
