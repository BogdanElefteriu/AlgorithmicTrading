from getData import download_binance_data

import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from scipy.optimize import minimize

raw_data = pd.DataFrame()
tickers = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
           'XRPUSDT', 'DOTUSDT', 'LUNAUSDT', 'AVAXUSDT', 'DOGEUSDT']

# for ticker in data:
#     download_binance_data(symbol= ticker, kline_size='1d', start_date= '1 Jan 2017')


# def getData(tickers):
#     raw_data = pd.DataFrame()
#     for ticker in tickers:
#         read_data = pd.read_csv('/Users/elefteriubogdan/PycharmProjects/TradingAlgo/data/raw/'+ticker+'_1d_raw_data.csv',
#                            index_col=0, header=0)
#         raw_data[ticker] = read_data[['close']]
#
#     raw_data = raw_data.dropna()
#     returns = raw_data.pct_change().dropna()
#     meanReturns = returns.mean()
#     covMatrix = returns.cov()
#     return raw_data, returns, meanReturns, covMatrix

# data, returns, meanRet, covMat = getData(tickers)
def actualPortfolioPerformance(weights, price):
    totalRet = 0
    for i in weights.iterrows():
        ret = ((price.iloc[-1][i[0]]-price.iloc[0][i[0]])/price.iloc[0][i[0]])*\
              (float(i[1]/100))
        totalRet += ret
    return totalRet

def annualisedPortfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*365   #anualised returns
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))*np.sqrt(365)
    return returns, std

def negativeSR(weights, meanReturns, covMatrix, riskFreeRate = 0):
    pReturns, pStd = annualisedPortfolioPerformance(weights, meanReturns, covMatrix)
    return -(pReturns-riskFreeRate)/pStd

def maxSR(meanReturns, covMatrix, riskFreeRate = 0, constraintSet = (0,1)):
    "Minimise the negative SR (Sharpe Ratio) by altering the weights/allocation of assets in the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq',
                    'fun': lambda x:np.sum(x)-1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))

    result = minimize(negativeSR, numAssets*[1./numAssets], args = args,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolioVariance(weights, meanReturn, covMatrix):
    return annualisedPortfolioPerformance(weights, meanReturn, covMatrix)[1]

def portfolioReturns(weights, meanReturn, covMatrix):
    return annualisedPortfolioPerformance(weights, meanReturn, covMatrix)[0]

def minVariance(meanReturns, covMatrix, constraintSet = (0,1)):
    "Minimise the portfolio variance by altering the weights/allocation of assets in the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq',
                    'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))

    result = minimize(portfolioVariance, numAssets * [1. / numAssets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficientFrontier(meanReturns, covMatrix, retTarget, constraintSet = (0,1)):
    """For each retTarget we want to optimise the portfolio for minimum variance """
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x:portfolioReturns(x, meanReturns, covMatrix)-retTarget},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effFront = minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return effFront

def calculatedResults(ret, meanReturns, covMatrix, riskFreeRate=0, contrainstSet=(0, 1)):
    """
    Read in mean, cov matrix, and other financial information
    Output, Max SR, Min Volatility, efficient frontier
    """
    # Max Sharpe Ratio Portfolio
    maxSR_Portofolio = maxSR(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = annualisedPortfolioPerformance(maxSR_Portofolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portofolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100.0) for i in maxSR_allocation.allocation] #transforms allocation in %

    # Min Variance (Volatility) Portfolio
    minVar_Portofolio = minVariance(meanReturns, covMatrix)
    minVar_returns, minVar_std = annualisedPortfolioPerformance(minVar_Portofolio['x'], meanReturns, covMatrix)
    minVar_allocation = pd.DataFrame(minVar_Portofolio['x'], index=meanReturns.index, columns=['allocation'])
    minVar_allocation.allocation = [round(i * 100.0) for i in minVar_allocation.allocation]  # transforms allocation in %

    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVar_returns, maxSR_returns, 20)
    for target in targetReturns:
        efficientList.append(efficientFrontier(meanReturns, covMatrix, target)['fun'])

    maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
    minVar_returns, minVar_std = round(minVar_returns*100,2), round(minVar_std*100,2)

    return maxSR_returns, maxSR_std, maxSR_allocation, minVar_returns, minVar_std, minVar_allocation, efficientList, targetReturns

def simulatePortfolios(returns, meanReturns, covMatrix, n_portfolios=2000, n_assets=10, filter=False):
    # -- Initialize empty list to store mean-variance pairs for plotting
    mean_variance_pairs = []
    weights_list = []
    tickers_list = []

    np.random.seed(75)
    # -- Loop through and generate lots of random portfolios
    for i in tqdm(range(n_portfolios)):
        next_i = False
        while True:
            # - Choose assets randomly without replacement
            assets = np.random.choice(list(returns.columns), n_assets, replace=False)
            # - Choose weights randomly
            weights = np.random.rand(n_assets)
            # - Ensure weights sum to 1
            weights = weights / sum(weights)

            # -- Loop over asset pairs and compute portfolio return and variance
            portfolio_Return = portfolioReturns(weights, meanReturns, covMatrix)
            portfolio_Variance = portfolioVariance(weights, meanReturns, covMatrix)

            # -- Skip over dominated portfolios
            if filter:
                for R, V in mean_variance_pairs:
                    if (R > round(portfolio_Return * 100, 2)) & (V < round(portfolio_Variance * 100, 2)):
                        next_i = True
                        break
                if next_i:
                    break

            # -- Add the mean/variance pairs to a list for plotting
            portfolio_Return = round(portfolio_Return * 100, 2)
            portfolio_Variance = round(portfolio_Variance * 100, 2)

            mean_variance_pairs.append([portfolio_Return, portfolio_Variance])
            weights_list.append(weights)
            tickers_list.append(assets)
            break

    return mean_variance_pairs, weights_list, tickers_list

def EF_plot(meanReturns, covMatrix, riskFreeRate=0, constraintSet = (0,1)):
    """Returns a plot of the min volatility, max sharpe ratio and efficient frontier"""
    calcSR_ret, calcSR_std, calcSR_allocation, \
    calcVar_ret, calcVar_std, calcVar_allocation, \
    efficientList, targetReturns = calculatedResults(meanRet, covMat)

    # Max SR
    maxSharpeRatio = go.Scatter(
        name='Maximum Sharpe Ratio',
        mode='markers',
        x=[calcSR_std],
        y=[calcSR_ret],
        marker=dict(color='red', size=14, line=dict(width=3, color='black')))

    # Min Var
    minVariance = go.Scatter(
        name='Minimum Volatility',
        mode='markers',
        x=[calcVar_std],
        y=[calcVar_ret],
        marker=dict(color='green', size=14, line=dict(width=3, color='black')))

    # Efficient Frontier
    effFront = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(eff_Std*100,2) for eff_Std in efficientList],
        y=[round(target*100,2) for target in targetReturns],
        line=dict(color='black', width=5, dash='dashdot'))

    # Generated Portfolios
    mean_variance_pairs, weights_list, tickers_list = simulatePortfolios(returns, meanReturns, covMatrix)
    mean_variance_pairs = np.array(mean_variance_pairs)
    simulations = go.Scatter(
        x=mean_variance_pairs[:, 1],
        y=mean_variance_pairs[:, 0],
        # - Add color scale for sharpe ratio
        marker=dict(color=(mean_variance_pairs[:, 0]) / (mean_variance_pairs[:, 1]),
                    showscale=True,
                    size=7,
                    line=dict(width=1),
                    colorscale="RdBu",
                    colorbar=dict(title="Sharpe<br>Ratio")),
        mode='markers',
        text=[str(np.array(tickers_list[i])) + "<br>" + str(np.array(weights_list[i]).round(2)) for i in range(len(tickers_list))])


    data = [maxSharpeRatio, minVariance, effFront, simulations]

    layout = go.Layout(
        title='Portfolio Optimisation with the Efficient Frontier',
        yaxis=dict(title='Annualised Return (%)'),
        xaxis=dict(title='Annualised Volatility (%)'),
        showlegend=True,
        legend=dict(x=0.75, y=0, traceorder='normal', bgcolor='#E2E2E2', bordercolor='black', borderwidth=2),
        width=800,
        height=600)

    fig = go.Figure(data=data, layout=layout)
    return fig.show()

# EF_plot(meanRet, covMat)
