# rag/news_pipeline.py
import yfinance as yf
import chromadb
import re
from datetime import datetime

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]

# --- TASK 1: Fetch news ---
def fetch_news(ticker: str) -> list[dict]:
    t = yf.Ticker(ticker)
    news = t.news
    results = []
    for item in news:
        results.append({
            "ticker": ticker,
            "title": item.get("title", ""),
            "summary": item.get("content", ""),   # yfinance key
            "published": str(item.get("providerPublishTime", "")),
            "url": item.get("link", ""),
        })
    return results

# --- TASK 2: Chunk text ---
def chunk_sentences(text: str, max_chars: int = 400) -> list[str]:
    sentences = re.split(f'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for a in sentences:
        if len(current) + len(a) <= max_chars:
            current += " " + a
        else:
            if a:
                chunks.append(current.strip())
            current = a
        
    if current:
        chunks.append(current.strip())

    return chunks

# --- TASK 3: Index into ChromaDB ---
def index_articles(collection, articles: list[dict]):
    for article in articles:
        text = str(article["title"]) + ". " + str(article["summary"])
        chunks = chunk_sentences(text)
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30:   # skip tiny chunks
                continue
            doc_id = f"{article['ticker']}_{article['published']}_{i}"
            collection.add(
                documents=[chunk],
                ids=[doc_id],
                metadatas=[{"ticker": article["ticker"], 
                            "url": article["url"], 
                            "published": int(datetime.fromisoformat(article["published"].replace("Z","")).timestamp()),
                            }]
            )

# --- TASK 4: Query ---
def query_ticker(collection, ticker: str, query: str, n: int = 3):
    results = collection.query(
        query_texts=[query],
        n_results=n,
        where={"ticker": ticker},
    )
    print(f"\n🔍 Query: '{query}' | Ticker: {ticker}")
    for doc in results["documents"][0]:
        print(f"  → {doc[:150]}...")

# --- MAIN ---
if __name__ == "__main__":
    client = chromadb.PersistentClient(path="./rag/chroma_db")
    # Use get_or_create so re-runs don't duplicate
    collection = client.get_or_create_collection("stock_news")
    
    for ticker in TICKERS:
        print(f"Fetching news for {ticker}...")
        articles = fetch_news(ticker)
        print(f"  {len(articles)} articles found")
        index_articles(collection, articles)
    
    print(f"\nTotal chunks indexed: {collection.count()}")
    
    # Test queries
    query_ticker(collection, "RELIANCE.NS", "quarterly earnings and profits")
    query_ticker(collection, "TCS.NS", "deal wins and revenue growth")
    query_ticker(collection, "HDFCBANK.NS", "loan growth and asset quality")