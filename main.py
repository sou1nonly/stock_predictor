from src.data.loader import DataLoader
from src.data.preprocessor import FeatureEngineer
from src.data.splitter import TimeSeriesSplitter
from src.models.trainer import ModelTrainer
from src.models.lstm_model import StockLSTM, LSTMTrainer
from src.models.versioner import Versioning
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
torch.manual_seed(43)
np.random.seed(55)


v = Versioning()

ticker = "AAPL"

# 1. Load data
loader = DataLoader(ticker, period="5y")
df = loader.yf_cleaned()

# 2. Feature engineering
f = FeatureEngineer(df)
f.to_returns()
f.acf_pacf_plot(lags=40)
featured_df = f.build()
print(f"Features shape: {featured_df.shape}")

# XGB training
#training = ModelTrainer(featured_df)
#training.train()
#training.predict()

# 3. Walk-forward validation
feature_cols = [c for c in featured_df.columns if c not in [ "Date", "Close", "Open", "High", "Low", "Returns"]]
x = featured_df[feature_cols].values
y = featured_df['Returns'].values
print(feature_cols)

# 4. Scale features 
split_raw = int(len(x) * 0.8)
scaler = StandardScaler()
X_train_raw = x[:split_raw]
X_test_raw = x[split_raw:]
X_train_scaled = scaler.fit_transform(X_train_raw)  # fit + transform on train
X_test_scaled = scaler.transform(X_test_raw)         # transform only on test (using train's mean/std)
X_scaled = np.vstack([X_train_scaled, X_test_scaled])

# 5. Create sequences 
seq_len = 30 

def create_sequences(x, y, seq_len):
    x_seq, y_seq = [], []

    for i in range(len(x) - seq_len):
        x_seq.append(x[i : i+seq_len])
        y_seq.append(y[i+seq_len])
    
    return np.array(x_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y, seq_len)

#6. Train Test Split
split = int(len(X_seq) * 0.8)
x_train, x_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

#7. Train the model
input_size = len(feature_cols)
lstm_model = StockLSTM(input_size=input_size)
trainer = LSTMTrainer(model=lstm_model,lr=0.001)
trainer.train(X_train=x_train, y_train=y_train, epochs=200)

#8. Predict 
predictions = trainer.predict(x_test)

#XGB trained

xgb_split_idx = split + seq_len
X_train_xgb = X_scaled[seq_len : xgb_split_idx]
y_train_xgb = y[seq_len : xgb_split_idx]
X_test_xgb = X_scaled[xgb_split_idx : ]
y_test_xgb = y[xgb_split_idx : ]
# Train and predict
xgb_model = XGBRegressor(n_estimators=150, random_state = 50, colsample_bytree= 0.4)
xgb_model.fit(X_train_xgb, y_train_xgb)
xgb_predictions = xgb_model.predict(X_test_xgb)

#feature importances
importance = xgb_model.feature_importances_
sorted_idx = np.argsort(importance)
plt.barh(np.array(feature_cols)[sorted_idx], importance[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# --- Calculate MAE ---
from sklearn.metrics import mean_absolute_error
lstm_mae = mean_absolute_error(y_test, predictions)
xgb_mae = mean_absolute_error(y_test_xgb, xgb_predictions)
print(f"\n--- Final Results ---")
print(f"LSTM MAE:    {lstm_mae:.4f}")
print(f"XGBoost MAE: {xgb_mae:.4f}")
# Plot both to compare
plt.figure(figsize=(12, 5))
plt.plot(y_test, label="Actual Returns", color="black", alpha=0.5)
plt.plot(predictions, label="LSTM Predicted", color="blue")
plt.plot(xgb_predictions, label="XGBoost Predicted", color="red", alpha=0.7)
plt.legend()
plt.title("LSTM vs XGBoost Predictions")
plt.savefig("LSTM vs XGB.png")
plt.show()

# Directional accuracy: did we get the sign right?
def directional_accuracy(y_true, y_pred):
    correct = np.sum(np.sign(y_true) == np.sign(y_pred))
    return correct / len(y_true)

lstm_dir = directional_accuracy(y_test, predictions)
xgb_dir = directional_accuracy(y_test_xgb, xgb_predictions)
print(f"LSTM Directional Accuracy: {lstm_dir:.2%}")
print(f"XGBoost Directional Accuracy: {xgb_dir:.2%}")
# Random guessing = 50%. Anything above 55% consistently is meaningful.

#saving models
metadata_XGB = {
    "version": "v1",
    "trained_at": f"{datetime.now()}",
    "ticker": "AAPL",
    "period": "5y",
    "features": feature_cols,         
    "n_features": len(feature_cols),
    "seq_len": 30,                   
    "xgb_directional_acc": xgb_dir,
    "xgb_mae": xgb_mae,
    "notes": "Added MACD, Bollinger, volume change, price range"
}
metadata_LSTM = {
    "version": "v1",
    "trained_at": f"{datetime.now()}",
    "ticker": "AAPL",
    "period": "5y",
    "features": feature_cols,   
    "n_features": len(feature_cols),
    "seq_len": 30,                    
    "lstm_directional_acc": lstm_dir,
    "lstm_mae": lstm_mae,
    "notes": "Added MACD, Bollinger, volume change, price range"
}

v.save_xgb(xgb_model, metadata_XGB, version="v1")
v.save_lstm(lstm_model, metadata_LSTM, version="v1")

v.list_versions()

from src.models.backtester import BackTester

bt = BackTester(initial_capital=10000)

xgb_results = bt.run(y_test_xgb, xgb_predictions)
print(f"XGBoost — Sharpe: {xgb_results['sharpe']:.2f}, Max Drawdown: {xgb_results['max_drawdown']:.2%}, Final Value: ${xgb_results['portfolio'][-1]:.2f}")
bt.plot(title="xgboost portfolio")

lstm_results = bt.run(y_test, predictions)
print(f"LSTM — Sharpe: {lstm_results['sharpe']:.2f}, Max Drawdown: {lstm_results['max_drawdown']:.2%}, Final Value: ${lstm_results['portfolio'][-1]:.2f}")
bt.plot(title="Lstm portfolio")