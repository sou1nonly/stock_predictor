# src/rag/retriever.py
import chromadb
import pandas as pd
import numpy as np
from typing import Optional

class NewsRetriever:
    def __init__(self, db_path: str = "./rag/chroma_db", collection_name: str = "stock_news"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)
    
    def retrieve(
        self, 
        ticker: str, 
        query: str, 
        n_results: int = 5,
        date_from: Optional[str] = None,
    ) -> list[dict]:
        """Return top-n relevant chunks. Optionally filter by date."""
        # TASK: Build the where filter
        # If date_from is provided, add {"published": {"$gte": date_from}}
        # Always filter by ticker
          # YOUR CODE
        
        if date_from is None:
            where_filter = {"ticker": ticker}
        else:
            where_filter = {
                "$and": [
                    {"ticker": ticker},
                    {"published": {"$gte": date_from}},
                ]
            }

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )
        
        # TASK: Parse results into list of dicts with keys: text, distance, metadata
        output = []  # YOUR CODE
        for i in range(len(results["documents"][0])):
            output.append({
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })
        return output
    
    def get_sentiment_context(self, ticker: str, n_results: int = 5) -> str:
        """Get recent news concatenated into one string for LLM consumption."""
        # TASK: Use self.retrieve() and join the text fields
        chunks = self.retrieve(ticker, "recent market news performance", n_results=5)
        return "\n---\n".join([c["text"] for c in chunks])
        pass
    
    @property
    def count(self) -> int:
        return self.collection.count()