from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
import os


class MultinomialNaiveBayesClassifier:
    def __init__(self, x_train, y_train):
        self._x_train = x_train
        self._y_train = y_train
        self._multinomial_naive_bayes = MultinomialNB()

    def train(self):
        self._multinomial_naive_bayes.fit(self._x_train, self._y_train)

    def test(self, x_test):
        return self._multinomial_naive_bayes.predict(x_test)

    def accuracy(self, x_test, y_test):
        return self._multinomial_naive_bayes.score(x_test, y_test)

    def get_average_f1_score(self, x_test, y_test):
        labels = [1, 0, -1]
        y_pred = self._multinomial_naive_bayes.predict(x_test)
        return f1_score(y_test, y_pred, average='weighted', labels=labels)


class BernoulliNaiveBayesClassifier:
    def __init__(self, x_train, y_train):
        self._x_train = x_train
        self._y_train = y_train
        self._bernoulli_naive_bayes = BernoulliNB()

    def train(self):
        self._bernoulli_naive_bayes.fit(self._x_train, self._y_train)

    def test(self, x_test):
        return self._bernoulli_naive_bayes.predict(x_test)

    def accuracy(self, x_test, y_test):
        return self._bernoulli_naive_bayes.score(x_test, y_test)

    def get_average_f1_score(self, x_test, y_test):
        labels = [1, 0, -1]
        y_pred = self._bernoulli_naive_bayes.predict(x_test)

        # Save predicted labels
        project_relative_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        print(project_relative_path)
        output_file_sentiment_label = open(os.path.join(project_relative_path, 'saved_model_data/naive_bayes_labels.txt'), 'a')
        for label in y_pred:
            output_file_sentiment_label.write(str(label))
            output_file_sentiment_label.write('\n')

        return f1_score(y_test, y_pred, average='weighted', labels=labels)


