import numpy as np
import pandas as pd
import math

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.closeprice = df["Close"].to_list()
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
                avg_gain.append(int(np.mean(gains[x-period+1: x+1])))
                avg_loss.append(int(np.mean(loss[x-period+1: x+1])))

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
    def cyclical_encode(value_col: list[int], max_value: int) -> list[int]:
        
        """ used for days and months to convert them into cyclic order of sin and cos """

        sin_val = math.sin(2 * math.pi * value_col / max_value)
        cos_val = math.cos(2 * math.pi *value_col / max_value)
        return sin_val, cos_val


    def build(self) -> pd.DataFrame:
        self.create_lag(1)
        self.create_lag(5)
        self.create_lag(30)
        self.rolling_mean(7)
        self.rolling_mean(30)
        self.compute_rsi()
   #     df = self.df[self.features]
        df = self.df.dropna()
        return df  