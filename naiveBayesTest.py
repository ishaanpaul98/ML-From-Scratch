import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

from naiveBayes import naiveBayes

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred) / len(y_true)
    return accuracy

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=1234)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

naiveBayesRegressor = naiveBayes()
naiveBayesRegressor.fit(X_train, y_train)
predicted = naiveBayesRegressor.predict(X_test)

print("Accuracy is: ", accuracy(y_test, predicted))