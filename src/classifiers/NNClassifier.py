from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class NearestNeighborClassifier:
    def __init__(self, x_train, y_train):
        self._x_train = x_train
        self._y_train = y_train
        self._nn_classifier = KNeighborsClassifier(n_neighbors=1)

    def train(self):
        self._nn_classifier.fit(self._x_train, self._y_train)

    def test_accuracy(self, x_test, y_test):
        y_prediction = self._nn_classifier.predict(x_test)
        return NearestNeighborClassifier._accuracy(y_prediction, y_test)

    @staticmethod
    def _accuracy(y_prediction, y_test):
        assert (len(y_prediction) == len(y_test))
        test_data_count = len(y_test)
        correct_classification_count = len([i for i, j in zip(y_prediction, y_test) if i == j])
        return correct_classification_count / test_data_count