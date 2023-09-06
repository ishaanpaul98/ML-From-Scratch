import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

from perceptron import Perceptron

def accuracy(y_true, y_pred):
    return (np.sum(y_true == y_pred) / len(y_true))

X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

perc = Perceptron()
perc.fit(X_train, y_train)
predicted = perc.predict(X_test)

print("Accuracy is: ", accuracy(y_test, predicted))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train)
x0_1 = np.amin(X_train[:,0])
x0_2 = np.amax(X_train[:,0])

x1_1 = (-perc.weights[0] * x0_1 - perc.bias / perc.weights[1])
x1_2 = (-perc.weights[0] * x0_2 - perc.bias / perc.weights[1])

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])

ax.set_ylim([ymin-3, ymax+3])

plt.show()