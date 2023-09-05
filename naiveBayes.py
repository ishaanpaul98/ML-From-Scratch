import numpy as np

class naiveBayes:
    def __init__(self):
        pass

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self._classes = np.unique(y)
        num_class = len(self._classes)
        #Init Mean, var, priors
        self._mean = np.zeros((num_class, num_features), dtype=np.float64)
        self._var = np.zeros((num_class, num_features), dtype=np.float64)
        self._priors = np.zeros(num_class, dtype=np.float64)

        for c in self._classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis = 0)
            self._var[c,:] = X_c.var(axis = 0)
            self._priors[c] = X_c.shape[0] / float(num_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self, x):
        posteriors = []
        for index, class_label in enumerate(self._classes):
            prior = self._priors[index]
            class_conditional = np.sum(np.log(self._probabilityDensityFunction(index, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _probabilityDensityFunction(self, class_index, x):
        mean = self._mean[class_index]
        var = self._var[class_index]
        numerator = np.exp(- (x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator