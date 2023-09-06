import numpy as np

class Perceptron:
    def __init__(self, lr = 0.01, num_iters=1000):
        self.lr = lr
        self.num_iters = num_iters
        self.activation_function = self._unitStepFunction
        self.bias = None
        self.weights = None

    def _unitStepFunction(self, x):
        return np.where(x>=0,1,0)
    
    def fit(self, X, y):
        num_samples, num_features = X.shape

        self.weights = np.zeros(num_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])
        for _ in range(self.num_iters):
            for index, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                update = self.lr * (y_[index] - y_predicted)
                self.weights += update * x_i
                self.bias += update


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted