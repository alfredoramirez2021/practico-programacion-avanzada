from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class ConsumptionModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

    def predict(self, X):
        return self.model.predict(X)
