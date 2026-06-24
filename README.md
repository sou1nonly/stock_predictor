# Stock Predictor

A machine learning-based stock price prediction API built with FastAPI, XGBoost, and yfinance.

## Overview

This project predicts stock price movements using historical data and machine learning. It provides a REST API to fetch stock data, generate features, train models, and make predictions.

## Features

- **Real-time Data Fetching**: Retrieve historical stock data using yfinance
- **Feature Engineering**: Automatic generation of lag features and technical indicators
- **Model Training**: XGBoost-based classification model for price movement prediction
- **REST API**: FastAPI-based endpoint for querying predictions
- **Health Check**: Built-in health monitoring endpoint

## Project Structure

```
.
├── api.py                      # FastAPI application and endpoints
├── stock_api.py                # Stock data fetching utilities
├── data_loader.py              # Data loading and preprocessing
├── feature_engineering.py       # Feature generation and transformation
├── model_training.py            # Model training and evaluation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock_predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- **fastapi**: Web framework for building APIs
- **uvicorn**: ASGI server
- **xgboost**: Gradient boosting machine learning library
- **yfinance**: Yahoo Finance API wrapper
- **pydantic**: Data validation using Python type hints
- **numpy**: Numerical computing

## Usage

### Running the API

Start the development server:
```bash
python api.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```
GET /health
```
Returns the API status and current timestamp.

#### Get Stock Data and Predictions
```
GET /stocks/{ticker}
```
Fetch historical data and predictions for a given stock ticker (e.g., AAPL, MSFT).

**Response Example:**
```json
{
  "ticker": "AAPL",
  "dates": ["2025-06-23", "2025-06-24"],
  "close": [150.25, 151.80],
  "prediction": true
}
```

### Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## How It Works

1. **Data Loading**: `DataLoader` fetches 1 year of historical price data
2. **Feature Engineering**: `FeatureEngineer` creates lag features for temporal patterns
3. **Model Training**: `ModelTrainer` trains an XGBoost classifier on historical data
4. **Prediction**: The API uses the trained model to predict price movements

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

Follow PEP 8 style guidelines. Format with black:
```bash
black .
```

## Performance Considerations

- Model predictions are based on 1 year of historical data
- Lag features are computed for recent price movements
- XGBoost handles non-linear relationships in stock price patterns

## Limitations

- Stock price prediction is inherently uncertain
- Model performance depends on market conditions and historical patterns
- Past performance does not guarantee future results
- Consider this as a reference tool, not financial advice

## Future Enhancements

- [ ] Add multiple stock ticker support
- [ ] Implement model caching for faster predictions
- [ ] Add confidence intervals to predictions
- [ ] Support for longer prediction horizons
- [ ] Integration with real-time streaming data
- [ ] Model persistence and versioning
- [ ] Unit and integration tests

## License

This project is open source and available under the MIT License.

## Disclaimer

This project is for educational purposes only. Stock market predictions are speculative and should not be used as sole basis for investment decisions. Always consult with a financial advisor before making investment decisions.
