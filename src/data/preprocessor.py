#feature_engineering.py
from typing import Self

import numpy as np
import pandas as pd
import math
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.closeprice = df["Close"].to_list()
        # To get month (1 to 12)
        self.df['Month'] = pd.to_datetime(self.df['Date']).dt.month
        # To get day of the week (0 to 6)
        self.df['DayOfWeek'] = pd.to_datetime(self.df['Date']).dt.dayofweek
    #    self.features = ["Close"]

    #lag1 features:
    def create_lag(self, lag: int) -> pd.DataFrame:
        final = []

        for x in range(len(self.closeprice)):
            if (x <= lag-1):
                final.append(np.nan)
            else:
                final.append(self.closeprice[x-lag])
        
        self.df[f"lag{lag}"] = final
        #self.features.append(f"lag{lag}")
        return self.df

    #rolling mean
    def rolling_mean(self, days: int) -> pd.DataFrame:
        final = []

        for i in range(len(self.closeprice)):
            if i < days - 1:
                final.append(np.nan)
            else:
                #final.append(sum(price[i-n+1: i+1])//n)
                final.append(np.mean(self.closeprice[i-days+1: i+1]))

        self.df[f"rolling{days}"] = final
        #self.features.append(f"rolling{days}")
        return self.df

    #compute RSI
    def compute_rsi(self, period:int = 14) -> pd.DataFrame:
        change = []

        for i in range(len(self.closeprice)):
            if i == 0:
                change.append(0)
            else:
                change.append(self.closeprice[i] - self.closeprice[i-1])
        #print(f"price {price}")
        #print(f"change{change}")
        
        gains = []
        loss = []

        for i in change:
            if i > 0:
                gains.append(abs(i))
                loss.append(0)
            elif(i == 0):
                gains.append(0)
                loss.append(0)
            else:
                loss.append(i)
                gains.append(0)

        #print(f"gains{gains}")
        #print(f"loss{loss}")

        avg_gain = [] #avg gain over n days
        avg_loss = [] #avg loss over n days


        for x in range(len(self.closeprice)):
            if(x < period-1):
                avg_gain.append(0)
                avg_loss.append(0)
            else:
                avg_gain.append(np.mean(gains[x-period+1: x+1]))
                avg_loss.append(np.mean(loss[x-period+1: x+1]))

        #print(f"avg gains {avg_gain}")
        #print(f"avg loss {avg_loss} ")
            
        rs = []
        RSI = []

        for x in range(len(avg_gain)):
    
            if avg_gain[x] > 0 and avg_loss[x] > 0:
                rs.append(avg_gain[x]/avg_loss[x])
            else:
                rs.append(avg_gain[x])
        #print(f"rs {rs}")
        for x in rs:
            final = 100-(100/(1+x))
            RSI.append(final)

        self.df[f"RSI{period}"] = RSI
        #self.features.append(f"RSI{period}")
        return self.df

    # cyclical_encode
    # pyrefly: ignore [invalid-annotation]
    def cyclical_encode(self, col_name:str , max_value: int) -> list[int]:
        
        """ used for days and months to convert them into cyclic order of sin and cos """

        self.df[f"{col_name}sin_val"] = np.sin(2 * np.pi * self.df[col_name] / max_value)
        self.df[f"{col_name}cos_val"] = np.cos(2 * np.pi * self.df[col_name] / max_value)

        return self.df

    def to_returns(self):

        self.df['Returns'] = self.df['Close'].pct_change()
        self.df = self.df.fillna(0)

        return self.df
    
    def check_stationary(self):
        result = adfuller(self.df['Returns'])
        p_value = result[1]

        return {"is_stationary": p_value < 0.05, "p_value": round(p_value, 4)}
    
    def acf_pacf_plot(self, lags):
        fig, axes = plt.subplots(1,2, figsize = (16, 4))

        plot_acf(self.df['Returns'], ax=axes[0], lags = lags)
        plot_pacf(self.df['Returns'], ax=axes[1], lags = lags)

        plt.savefig("acf_pacf.png")
        plt.close()

    def addMACD(self):
        ema_12  = self.df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.df['Close'].ewm(span=26, adjust= False).mean()
        self.df['MACD'] = ema_26 - ema_12
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

        return self.df
    
    def add_bollinger_width(self, window=20):
        rollingmean = self.df['Close'].rolling(window=window).mean()
        rollingstd = self.df['Close'].rolling(window=window).std()
        self.df['bb_width'] = (2* rollingstd) / rollingmean

        return self.df
    
    def add_volume_change(self):
        self.df['Volume_change'] = self.df['Volume'].pct_change()

        return self.df
    
    def add_price_range(self):
        self.df['Price_range'] = self.df['High'] - self.df['Low']
        self.df['Price_range'] = self.df['Price_range']/ self.df['Close']

        return self.df


    def build(self) -> pd.DataFrame:
        self.create_lag(1)
        self.create_lag(5)
        self.create_lag(20)
        self.rolling_mean(15)
        self.rolling_mean(30)
        self.compute_rsi()
        #self.cyclical_encode(col_name='Month', max_value=12)
        #self.cyclical_encode(col_name='DayOfWeek', max_value=7)
        self.addMACD()
        self.add_bollinger_width(window=20)
        self.add_volume_change()
        self.add_price_range()
   #    df = self.df[self.features]
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = self.df.dropna()
        return df  
    
    
