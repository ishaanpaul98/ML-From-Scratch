import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

#print(X_train.shape)
#print(X_train[0])

#print(y_train.shape)
#print(y_train)

#plt.figure()
#plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap, edgecolor = 'k', s = 20)
#plt.show()

from KNN import KNN
knn = KNN(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = np.sum(predictions==y_test) / len(y_test)
print(accuracy)