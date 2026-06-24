# Your imports
import yfinance as yf
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import math
from datetime import datetime, timedelta
from xgboost import XGBClassifier

app = FastAPI(title="Stock Predictor API")

def yf_cleaned(ticker:str, p:str ):

    stock = yf.download(ticker, period= p)
    stock = stock.droplevel('Ticker', axis=1)  
    stock = stock.reset_index()

    return stock

#lag features
def create_lag(prices, lag):
    final = []

    for x in range(len(prices)):
        if (x <= lag-1):
            final.append(np.nan)
        else:
            final.append(prices[x-lag])
    return final

#rolling mean
def rolling_mean(price, n):
    final = []
    for i in range(len(price)):
        if i < n - 1:
            final.append(np.nan)
        else:
            #final.append(sum(price[i-n+1: i+1])//n)
            final.append(np.mean(price[i-n+1: i+1]))

    return final

#compute RSI
def compute_rsi(price, period):
    change = []

    for i in range(len(price)):
        if i == 0:
            change.append(0)
        else:
            change.append(price[i] - price[i-1])
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


    for x in range(len(price)):
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

    return RSI

# cyclical_encode
def cyclical_encode(value, max_value):
    sin_val = math.sin(2 * math.pi * value / max_value)
    cos_val = math.cos(2 * math.pi *value / max_value)
    return sin_val, cos_val

#for day in range(7):
    #s,c = cyclical_encode(day,7)
    #print(f"day {day}: sin={s:.3f} cos={c:.3f}")


@app.get("/health")
def health():
    return{"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/stocks/{ticker}")
def stock_data(ticker: str):
    df = yf_cleaned(ticker, p="1y")
    dates = list(df["Date"])
    starting_date = dates[0]
    ending_date = dates[-1]
    close = df["Close"].to_list()
    close = close[-1]

    if df.empty:
        return {"ticker": ticker, "last close price":close, "date range": f"{starting_date} - {ending_date}"}
    else:
        raise HTTPException(status_code=404, detail="ticker not found")

class predictions(BaseModel):
    ticker: str
    period: str = "1y"

@app.post("/predict")
def predict(p: predictions):
    try:
        df = yf_cleaned(p.ticker, p.period)
        closeprice = list(df["Close"])
        df["lag1"] = create_lag(closeprice, 7)
        df["lag7"] = create_lag(closeprice, 30)
        df["rolling7"] = rolling_mean(closeprice, 7)
        df["rolling30"] = rolling_mean(closeprice, 30)
        df["rsi"] = compute_rsi(closeprice, 14)

        features = ['lag1', 'lag7', 'rolling7', 'rolling30', 'rsi']
        df['target'] = (df['Close'].shift() > df['Close']).astype(int)
        df = df.dropna()

        X = df[features]
        y = df["target"]

        test_case = X.iloc[-1:]
        X = X.iloc[:-1]
        y = y.iloc[:-1]

        split = int(len(X)*0.7)

        X_train = X[: split]
        y_train = y[: split]
        X_test = X[split:]
        y_test = y[split:]

        model = XGBClassifier(n_estimators = 150, random_state = 40)
        model.fit(X_train,y_train)
        accuracy = model.score(X_test, y_test)
        
        testpred = model.predict(test_case)
        pred = "something wrong"

        if (testpred == 0):
            pred = "down"
        else: 
            pred = "up"

        return { "ticker": p.ticker, "prediction": pred, "confidence": accuracy, "number of data points": len(df)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail= f"prediction falied: {str(e)}")
    
@app.get("/stocks/{ticker}/features")
def feature(ticker:str):
    try:
        df = yf_cleaned(ticker, "1y")
        closeprice = list(df["Close"])
        df["lag1"] = create_lag(closeprice, 1)
        df["lag7"] = create_lag(closeprice, 30)
        df["rolling7"] = rolling_mean(closeprice, 7)
        df["rolling30"] = rolling_mean(closeprice, 30)
        df["rsi"] = compute_rsi(closeprice, 14)

        features = ['lag1', 'lag7', 'rolling7', 'rolling30', 'rsi']
        df['target'] = (df['Close'].shift() > df['Close']).astype(int)
        df = df.dropna()

        return {"features": df.tail(5).to_dict(orient="re")}

    except Exception as e:
        raise HTTPException(status_code=500, detail= f"details not found {str(e)}")    


if __name__== "__main__":
    uvicorn.run(app, host="localhost", port= 8010 )
