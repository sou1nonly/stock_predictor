#data_loader.py
import pandas as pd
import yfinance as yf

class DataLoader:
    def __init__(self, ticker:str, period:str = "1y"):
        self.ticker = ticker
        self.period = period

    def yf_cleaned(self) -> pd.DataFrame:

        stock = yf.download(tickers = self.ticker, period = self.period)

        if stock.empty:
            return stock
        
        stock = stock.droplevel('Ticker', axis=1)  
        stock = stock.reset_index()

        return stock
            