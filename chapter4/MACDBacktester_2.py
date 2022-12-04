
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")

class MACDBacktester_2(): 
    ''' Class for the vectorized backtesting of MACD-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    EMA_S: int
        time window in days for shorter EMA
    EMA_L: int
        time window in days for longer EMA
    signal_mw: int
        time window is days for MACD Signal 
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
        sets new MACD parameter(s)
        
    test_strategy:
        runs the backtest for the MACD-based strategy
        
    plot_results:
        plots the performance of the strategy compared to buy and hold
        
    update_and_run:
        updates MACD parameters and returns the negative absolute performance (for minimization algorithm)
        
    optimize_parameters:
        implements a brute force optimization for the three MACD parameters
    '''
    
    def __init__(self, symbol, EMA_S, EMA_L, signal_mw, start, end, tc):
        self.symbol = symbol
        self.EMA_S = EMA_S
        self.EMA_L = EMA_L
        self.signal_mw = signal_mw
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None 
        self.get_data()
        
    def __repr__(self):
        return "MACDBacktester(symbol = {}, MACD({}, {}, {}), start = {}, end = {})".format(self.symbol, self.EMA_S, self.EMA_L, self.signal_mw, self.start, self.end)
        
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        df = yf.download(self.symbol, start = self.start, end = self.end)
        
        df.bfill()
        
        df.drop(df.columns.difference(['Adj Close']), 1, inplace=True)
        
        df.columns = ["price"]
        
        df["returns"] = np.log(df / df.shift(1))
        df["EMA_S"] = df["price"].ewm(span = self.EMA_S, min_periods = self.EMA_S).mean() 
        df["EMA_L"] = df["price"].ewm(span = self.EMA_L, min_periods = self.EMA_L).mean()
        df["MACD"] = df.EMA_S - df.EMA_L
        df["MACD_Signal"] = df.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean() 
        self.data = df
      
        
    def set_parameters(self, EMA_S = None, EMA_L = None, signal_mw = None):
        ''' Updates MACD parameters and resp. time series.
        '''
        if EMA_S is not None:
            self.EMA_S = EMA_S
            self.data["EMA_S"] = self.data["price"].ewm(span = self.EMA_S, min_periods = self.EMA_S).mean()
            self.data["MACD"] = self.data.EMA_S - self.data.EMA_L
            self.data["MACD_Signal"] = self.data.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean()
            
        if EMA_L is not None:
            self.EMA_L = EMA_L
            self.data["EMA_L"] = self.data["price"].ewm(span = self.EMA_L, min_periods = self.EMA_L).mean()
            self.data["MACD"] = self.data.EMA_S - self.data.EMA_L
            self.data["MACD_Signal"] = self.data.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean()
            
        if signal_mw is not None:
            self.signal_mw = signal_mw
            self.data["MACD_Signal"] = self.data.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean()
            
    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data["position"] = np.where(data["MACD"] > data["MACD_Signal"], 1, -1)
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
        
        
        print("parameters: " + str(self.EMA_S) +", "+ str(self.EMA_L)+", "+str(self.signal_mw))
    
        return perf, benchmark, drawdown_strategy, drawdown_BH, data
    
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            plt.rc('legend',fontsize=20) # using a size in points
            title = "S&P 500 Index | MACD ({}, {}, {}) | TC = {}".format(self.EMA_S, self.EMA_L, self.signal_mw, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
        
    def update_and_run(self, MACD):
        ''' Updates MACD parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========                    
        MACD: tuple
            MACD parameter tuple
        '''
        self.set_parameters(int(MACD[0]), int(MACD[1]), int(MACD[2]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, EMA_S_range, EMA_L_range, signal_mw_range):
        ''' Finds global maximum given the MACD parameter ranges.

        Parameters
        ==========
        EMA_S_range, EMA_L_range, signal_mw_range : tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (EMA_S_range, EMA_L_range, signal_mw_range), finish=None)
        return opt, -self.update_and_run(opt)

    
