import numpy as np
import matplotlib.pyplot as plt

class BackTester():
    def __init__(self, initial_capital = 10000):
        self.initial_capital = initial_capital
        self.capital_change = [self.initial_capital]

    def run(self, actual_returns, predicted_returns, risk_free = 0):
        self.capital_change = [self.initial_capital]
        remaining = []
        strat_returns = 0

        for i, j in zip(actual_returns, predicted_returns):
            if np.sign(i) == np.sign(j):
                remaining.append(abs(j))

            else:
                remaining.append(-abs(j))

        for i in remaining:
            self.capital_change.append(self.capital_change[-1] * (1 + i))
        
        excess = np.array(remaining) - risk_free
        if np.std(excess) == 0:
            strat_returns = 0
        else: 
            strat_returns = np.mean(excess) / np.std(excess) * np.sqrt(252)
        
        peak = self.capital_change[0]
        max_dd = 0

        for i in self.capital_change:
            if i > peak:
                peak = i
            dd = (peak - i) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            "total returns" : (self.capital_change[-1] - self.initial_capital) / self.initial_capital,
            "sharpe": strat_returns,
            "max_drawdown": max_dd,
            "portfolio": self.capital_change
        }

    def plot(self, title= "portfolio over time"):
        plt.figure()
        plt.title(title)
        plt.plot(self.capital_change)
        plt.ylabel("portfolio value ")
        plt.xlabel("trading days")
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()


