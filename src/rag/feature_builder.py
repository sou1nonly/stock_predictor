# src/rag/feature_builder.py
import pandas as pd
import numpy as np
from datetime import datetime
from src.rag.retriever import NewsRetriever
from src.rag.sentiment import SentimentAnalyzer

class SentimentFeatureBuilder:
    """Produces a daily sentiment feature series for a given ticker."""
    
    def __init__(self, api_key: str, db_path: str = "./rag/chroma_db"):
        self.retriever = NewsRetriever(db_path=db_path)
        self.analyzer = SentimentAnalyzer(api_key=api_key)
    
    def get_today_sentiment(self, ticker: str) -> dict:
        """Get today's sentiment score + metadata for a ticker."""
        context = self.retriever.get_sentiment_context(ticker)
        result = self.analyzer.analyze(ticker, context)
        
        if result is None:
            return {
                "ticker": ticker,
                "date": datetime.today().strftime("%Y-%m-%d"),
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "sentiment_weighted": 0.0,
                "summary": "No data",
            }
        
        return {
            "ticker": ticker,
            "date": datetime.today().strftime("%Y-%m-%d"),
            "sentiment_score": result.sentiment_score,
            "confidence": result.confidence,
            "sentiment_weighted": result.sentiment_score * result.confidence,  # Approach B
            "summary": result.summary,
        }
    
    def build_sentiment_row(self, featured_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add sentiment features to an existing featured_df.
        
        TASK: 
        1. Call get_today_sentiment(ticker)
        2. Add sentiment_score, confidence, sentiment_weighted columns to featured_df
        3. Apply shift(1) to align news from today → tomorrow's prediction
        4. Return the modified df
        """
        today = self.get_today_sentiment(ticker)
        featured_df["sentiment_score"] = today.sentiment_score
        featured_df["confidence"] = today.confidence
        featured_df[["sentiment_score", "confidence"]] = featured_df[["sentiment_score", "confidence"]].shift(1)
        featured_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = featured_df.dropna()

        return df