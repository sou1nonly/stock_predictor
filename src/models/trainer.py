#model_training.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
logging.basicConfig(level=logging.INFO)
# add to main.py after trainer.evaluate(...)
logger = logging.getLogger(__name__)


class ModelTrainer():
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
        split = int(len(self.df)*0.7)

        self.training = self.df.iloc[:split]
        self.testing = self.df.iloc[split:]

        #regression
        feature_cols = [c for c in self.df.columns if c not in ["Date", "Close", "Open", "High", "Low"]]
        target_cols = "Close"

        print(feature_cols)

        self.X_train, self.y_train = self.training[feature_cols], self.training[target_cols]
        X_test, y_test = self.testing[feature_cols], self.testing[target_cols]
        
        self.test_case = X_test.iloc[-1:]
        self.X_test = X_test[:-1]
        self.y_test = y_test[:-1]

        self.model = XGBRegressor(n_estimators = 100, random_state = 42)


    def train(self) -> float:

        self.model.fit(self.X_train,self.y_train)
        accuracy = self.model.score(self.X_test, self.y_test)
        print(f"accuracy: {accuracy}")
        return {"accuracy": accuracy}
    
    def evaluate(self):
        preds = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, preds)
        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        r2 = r2_score(self.y_test, preds)
        
        print("\n--- Evaluation Results ---")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")
        return {"mae": mae, "rmse": rmse, "r2": r2}
    
    def predict(self) -> str :
        pred = self.model.predict(self.test_case)

        return pred

    def predict_accuracy_graph(self):
        predictions = list(self.model.predict(self.X_test))
        sns.lineplot(data=predictions)
        sns.lineplot(data=self.y_test)
        plt.plot()

    def actual_predicted(self):
        preds = self.model.predict(self.X_test)
        plt.figure(figsize=(12, 5))
        plt.plot(self.testing["Date"].iloc[:-1].values, self.y_test.values, label="Actual", color="steelblue")
        plt.plot(self.testing["Date"].iloc[:-1].values, preds, label="Predicted", color="orange", linestyle="--")
        plt.title(f" Actual vs Predicted Close Price")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("acc_plot.png")
        print("[5] Plot saved as prediction_plot.png")


    