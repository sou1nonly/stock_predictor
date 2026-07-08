import pandas as pd

class TimeSeriesSplitter():
    def __init__(self, n_splits, gap, test_size):
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size

        self.train_indices = []
        self.test_indices = []

    
    def split(self, df: pd.DataFrame):

        initial_train_size = len(df) - (self.n_splits * self.test_size)

        for i in range(self.n_splits):
           train_end = initial_train_size + (i * self.test_size)
           test_start = train_end + self.gap
           test_end = test_start + self.test_size

           self.train_indices.append(train_end)
           self.test_indices.append(test_end)

        return self.train_indices, self.test_indices
    
    def validate(self, df:pd.DataFrame, model, metric_fn, feature_cols:pd.array, target_cols:str):

        acc = []

        for x in range(self.n_splits):

            train = df.iloc[:self.train_indices[x]]
            test = df.iloc[self.train_indices[x]+self.gap: self.test_indices[x]]

            X_train, X_test = train[feature_cols], test[feature_cols]
            y_train, y_test = train[target_cols], test[target_cols]
                
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metric = metric_fn(y_test, pred)
            acc.append(metric)
        
        return acc

            




        