import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer():
    def __init__(self, df: pd.DataFrame):
        self.df = df

        df["target"] = (df["Close"].shift() > df["Close"]).astype(int)
        y = df["target"]
        x = df.drop(columns=["Close"])

        logger.info(f"the type of x{type(x)}")
        logger.info(f"the type of y {type(y)}")

        self.test_case = x.iloc[-1:]
        self.x = x.iloc[:-1]
        self.y = y.iloc[:-1]

        split = int(len(self.x)*0.7)
        self.X_train = self.x[: split]
        self.y_train = self.y[: split]
        self.X_test = self.x[split:]
        self.y_test = self.y[split:]

        logger.info(f"the type of x{type(self.x)}")
        logger.info(f"the type of y {type(self.y)}")

        self.model =  XGBClassifier(n_estimators = 150, random_state = 40)


    def train(self) -> float:

        self.model.fit(self.X_train,self.y_train)
        accuracy = self.model.score(self.X_test, self.y_test)

        return accuracy
    
    def predict(self) -> str :
        testpred = self.model.predict(self.test_case)
        pred = " "

        if (testpred <= 0):
            pred = "down"
        else: 
            pred = "up"

        return pred

    def predict_accuracy_graph(self):
        predictions = list(self.model.predict(self.X_test))
        sns.lineplot(data=predictions)
        sns.lineplot(data=self.y_test)
        plt.plot()

    


    