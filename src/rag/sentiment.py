# src/rag/sentiment.py
import json
from dataclasses import dataclass, asdict
from google.genai import types
from typing import Optional
import google.genai as genai

# You'll need: pip install google-generativeai
# Get a free API key from https://aistudio.google.com/apikey

SENTIMENT_PROMPT = """You are a senior financial analyst specializing in Indian equities.
Analyze the following news articles about {ticker} and provide a structured assessment.

NEWS ARTICLES:
{context}

Respond in EXACTLY this JSON format, nothing else:
{{
  "sentiment_score": <float from -1.0 to 1.0>,
  "confidence": <float from 0.0 to 1.0>,
  "key_drivers": ["<driver1>", "<driver2>", "<driver3>"],
  "summary": "<2-3 sentence summary>",
  "risk_factors": ["<risk1>", "<risk2>"]
}}

Scoring guide:
  -1.0 = catastrophic news (fraud, bankruptcy)
  -0.5 = clearly negative (earnings miss, downgrades)
   0.0 = neutral or genuinely mixed signals
  +0.5 = clearly positive (earnings beat, upgrades)
  +1.0 = exceptional (breakthrough deal, major expansion)

If articles are insufficient, set confidence to 0.0 and sentiment_score to 0.0.
Do NOT hallucinate information not present in the articles.
"""

@dataclass
class SentimentResult:
    ticker: str
    sentiment_score: float
    confidence: float
    key_drivers: list
    summary: str
    risk_factors: list
    raw_response: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_sentiment(ticker: str, raw: str) -> Optional[SentimentResult]:
    """Parse and validate LLM JSON response."""
    # TASK: Implement this
    # 1. Try json.loads(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[warn] Invalid json for {ticker}: {raw[:200]}")

    # 2. Check required fields exist
    required = ["sentiment_score", "confidence", "key_drivers", "summary"]
    for f in required:
        if f not in data:
            print(f"[warn] Missing Field '{f}' for {ticker}")
            return None
        
     # 3. Clamp score to [-1, 1] and confidence to [0, 1]
    score = max(-1.0, min(1.0, float(data["sentiment_score"])))
    conf = max(0.0, min(1.0, float(data["confidence"])))

    # 4. Return SentimentResult or None
    return SentimentResult(
        ticker= ticker,
        sentiment_score=score,
        confidence=conf,
        key_drivers=data.get("key_drivers", [])[:5],
        summary=data.get("summary", "No summary available."),
        risk_factors=data.get("risk_factors", []),
        raw_response=raw,
    )


class SentimentAnalyzer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def analyze(self, ticker: str, context: str) -> Optional[SentimentResult]:
        if len(context.strip()) < 50:
            return None

        prompt = SENTIMENT_PROMPT.format(ticker=ticker, context=context)

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        result = parse_sentiment(ticker, response.text)
        if result and result.confidence < 0.3:
            return None
        return result