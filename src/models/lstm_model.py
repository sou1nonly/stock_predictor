import torch
import torch.nn as nn
torch.manual_seed(42)
from torch.utils.data import Dataset, DataLoader

class StockLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, batch_first=True)
        self.linear = nn.Linear(32, 1)

    def forward(self, x ):
        output, (hidden, cell) = self.lstm(x)

        last = output[:, -1, :]
        pred = self.linear(last)

        return pred
    
class LSTMTrainer():
    def __init__(self, model, lr = 0.001):
        self.model = model
        self.lr = lr
        self.optimiser = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criteria = nn.MSELoss()

    def train(self, X_train, y_train, epochs = 50):
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)

        losses = []

        for i in range(epochs):
            self.optimiser.zero_grad()
            predictions = self.model(X_train)
            loss = self.criteria(predictions.squeeze(), y_train)
            loss.backward()
            self.optimiser.step()
            
            if i % 10 == 0:
                print(f"epoch {i} loss = {loss.item()}")

            losses.append(loss.item())

        return losses

    def predict(self, x_test):
        self.model.eval()

        x_test = torch.FloatTensor(x_test)
        
        with torch.no_grad():
            predictions = self.model(x_test)

        return predictions.squeeze().numpy()
    

