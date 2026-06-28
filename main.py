from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
import matplotlib.pyplot as plt


ticker = "AAPL"
time = "3y"

# load data
loader = DataLoader(ticker, time)
df = loader.yf_cleaned()
print(f"loaded {len(df)} rows for {ticker}")

# feature engineering
f = FeatureEngineer(df)
featured_df = f.build()
print(f"features created {featured_df.head(4)}")

# train/ test model
trained = ModelTrainer(featured_df)
trained.train()
trained.evaluate()
trained.actual_predicted()
