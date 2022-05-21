from getDataGecko import getData
from portolioAnalysis import calculatedResults, actualPortfolioPerformance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

coins = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']

# 'XRP', 'DOT', 'LUNA', 'AVAX', 'DOGE'

coins_name = ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano']
              # 'ripple', 'polkadot', 'terra-luna', 'avalanche-2', 'dogecoin'

def splitData(data):
    price = pd.DataFrame(index=data[0].index, columns=coins)
    market_cap = pd.DataFrame(index=data[0].index, columns=coins)
    supply = pd.DataFrame(index=data[0].index, columns=coins)
    i = 0
    for coin in data:
        price.iloc[:, i] = coin['price']
        market_cap.iloc[:, i] = coin['market_cap']
        supply.iloc[:, i] = coin['coin_supply']
        i += 1
    return price, market_cap, supply

def processData(raw_data):
    returns = raw_data.pct_change().dropna()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix


# Import price, market cap and supply of each coin
data = getData(coins_name)

price, market_cap, supply = splitData(data)

returns, meanRet, covMat = processData(price)


# Generate weights (Efficient Frontier method) based on the portfolio
calcSR_ret, calcSR_std, calcSR_allocation, \
calcVar_ret, calcVar_std, calcVar_allocation, \
efficientList, targetReturns = calculatedResults(returns, meanRet, covMat)

actualSR_ret = actualPortfolioPerformance(calcSR_allocation, price)

def computeIndex(price, market_cap, weights):
    index = pd.DataFrame(columns=['Index'], index=price.index)
    index_count = 0
    for time in range(len(price)):
        for weight in weights.iterrows():
            value = (float(weight[1]/100) * market_cap.iloc[time][weight[0]])
            index_count += value
        index.loc[price.index[time]] = index_count
        index_count = 0
    return index


# indexDivisor = 4.166956878*1e+8

index = computeIndex(price, market_cap, calcSR_allocation)
index = index/index.iloc[0,0]

plt.plot(index)
plt.show()
