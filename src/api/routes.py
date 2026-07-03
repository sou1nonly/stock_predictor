from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from src.data.loader import DataLoader
from src.data.preprocessor import FeatureEngineer
from src.models.trainer import ModelTrainer


app = FastAPI(title="Stock Predictor API")

@app.get("/health")
def health():
    return{"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/stocks/{ticker}")
def stock_data(ticker: str):
    loader = DataLoader(ticker = ticker, period = "1y")
    df = loader.yf_cleaned()

    dates = df["Date"].to_list()
    close = df["Close"].to_list()

    if df.empty:
        raise HTTPException(status_code=404, detail="ticker not found")
    else:
        return {"ticker": ticker, "last close price": close[-1], "date range": f"{dates[0]} - {dates[-1]}"}


class PredictionRequest(BaseModel):
    ticker: str
    period: str = "1y"

@app.post("/predict")
def predict(p: PredictionRequest):
    try:
        loader = DataLoader(p.ticker, p.period)
        df_loaded = loader.yf_cleaned()

        featured = FeatureEngineer(df=df_loaded)
        df_featured = featured.build()

        training = ModelTrainer(df_featured)
        accuracy = training.train()
        pred = training.predict()
        training.predict_accuracy_graph()

        return { "ticker": p.ticker, "prediction": pred, "accuracy": accuracy, "number of data points": df_loaded.shape}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail= f"prediction falied: {str(e)}")
    
@app.get("/stocks/{ticker}/features")
def feature(ticker:str):
    try:
        loader = DataLoader(ticker=ticker)
        df_loaded = loader.yf_cleaned()

        featured = FeatureEngineer(df=df_loaded)
        df_featured = featured.build()

        return {"features": df_featured.tail(5).to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail= f"details not found {str(e)}")    


if __name__== "__main__":
    uvicorn.run(app, host="localhost", port= 8015 )
