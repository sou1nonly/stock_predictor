from src.data.loader import DataLoader
from src.data.preprocessor import FeatureEngineer
from src.data.splitter import TimeSeriesSplitter
from src.models.trainer import ModelTrainer
from src.models.lstm_model import StockLSTM, LSTMTrainer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

ticker = "AAPL"

# 1. Load data
loader = DataLoader(ticker, period="10y")
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

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
split = int(len(X_seq) * 0.75)
x_train, x_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

#7. Train the model
input_size = len(feature_cols)
model = StockLSTM(input_size=input_size)
trainer = LSTMTrainer(model=model,lr=0.001)
trainer.train(X_train=x_train, y_train=y_train, epochs=50)

#8. Predict 
predictions = trainer.predict(x_test)

#9. Results
plt.plot(y_test)
plt.plot(predictions)
plt.savefig('actual vs predicted')
plt.close()


xgb_split_idx = split + seq_len
X_train_xgb = X_scaled[seq_len : xgb_split_idx]
y_train_xgb = y[seq_len : xgb_split_idx]
X_test_xgb = X_scaled[xgb_split_idx : ]
y_test_xgb = y[xgb_split_idx : ]
# Train and predict
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train_xgb, y_train_xgb)
xgb_predictions = xgb_model.predict(X_test_xgb)

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