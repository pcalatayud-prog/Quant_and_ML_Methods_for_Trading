
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
import yfinance as yf
plt.style.use("seaborn")

class SOBacktester_2(): 
    ''' Class for the vectorized backtesting of SO-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    periods: int
        time window in days for rolling low/high
    D_mw: int
        time window in days for %D line
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    tc: float
        proportional transaction costs per trade
        
        
    Methods
    =======
    get_data:
        retrieves and prepares the data
        
    set_parameters:
        sets one or two new SO parameters
        
    test_strategy:
        runs the backtest for the SO-based strategy
        
    plot_results:
        plots the performance of the strategy compared to buy and hold
        
    update_and_run:
        updates SO parameters and returns the negative absolute performance (for minimization algorithm)
        
    optimize_parameters:
        implements a brute force optimization for the two SO parameters
    '''
    
    def __init__(self, symbol, periods, D_mw, start, end, tc):
        self.symbol = symbol
        self.periods = periods
        self.D_mw = D_mw
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None 
        self.get_data()
        
    def __repr__(self):
        return "SOBacktester(symbol = {}, periods = {}, D_mw = {}, start = {}, end = {})".format(self.symbol, self.periods, self.D_mw, self.start, self.end)
        
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        df = yf.download(self.symbol, start = self.start, end = self.end)
        
        df.bfill()
    
        
        df["returns"] = np.log(df.Close / df.Close.shift(1))
        df["roll_low"] = df.Low.rolling(self.periods).min()
        df["roll_high"] = df.High.rolling(self.periods).max()
        df["K"] = (df.Close - df.roll_low) / (df.roll_high - df.roll_low) * 100
        df["D"] = df.K.rolling(self.D_mw).mean()
        self.data = df
        
    def set_parameters(self, periods = None, D_mw = None):
        ''' Updates SO parameters and resp. time series.
        '''
        if periods is not None:
            self.periods = periods
            self.data["roll_low"] = self.data.Low.rolling(self.periods).min()
            self.data["roll_high"] = self.data.High.rolling(self.periods).max()
            self.data["K"] = (self.data.Close - self.data.roll_low) / (self.data.roll_high - self.data.roll_low) * 100
            self.data["D"] = self.data.K.rolling(self.D_mw).mean() 
        if D_mw is not None:
            self.D_mw = D_mw
            self.data["D"] = self.data.K.rolling(self.D_mw).mean()
            
    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data["position"] = np.where(data["K"] > data["D"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        # determine when a trade takes place
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # subtract transaction costs from return when trade takes place
        data.strategy = data.strategy - data.trades * self.tc
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        
        #Calculating drawdown
        data["cummax_BH"] = data.creturns.cummax()
        data["cummax_strategy"] = data.cstrategy.cummax()
        
        data["drawndown_BH"] = (data["cummax_BH"] - data["creturns"])/data["cummax_BH"]
        data["drawndown_strategy"] = (data["cummax_strategy"] - data["cstrategy"])/data["cummax_strategy"]
        
        self.results = data
        
        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        perf = round(perf, 3)
        
        benchmark = data["creturns"].iloc[-1] #Benchmark - buy & hold strategy
        benchmark = round(benchmark, 3)
        
        outperf = perf - benchmark # out-/underperformance of strategy
        
        drawdown_BH = round(data["drawndown_BH"].max(),3)
        drawdown_strategy = round(data["drawndown_strategy"].max(),3)
        
        print("Strategy Performance: " + str(perf))
        print("Hold and Buy Performance: " + str(benchmark))
        print("Strategy Maximun Drawdown: " + str(drawdown_strategy))
        print("Hold and Buy Drawdown: " + str(drawdown_BH))
        
        
        print("parameters: " + str(self.periods) +", "+ str(self.D_mw))
    
        return perf, benchmark, drawdown_strategy, drawdown_BH, data
    
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | periods = {}, D_mw = {} | TC = {}".format(self.symbol, self.periods, self.D_mw, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
        
    def update_and_run(self, SO):
        ''' Updates SO parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        SO: tuple
            SO parameter tuple
        '''
        self.set_parameters(int(SO[0]), int(SO[1]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, periods_range, D_mw_range):
        ''' Finds global maximum given the SO parameter ranges.

        Parameters
        ==========
        periods_range, D_mw_range: tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (periods_range, D_mw_range), finish=None)
        return opt, -self.update_and_run(opt)

    
