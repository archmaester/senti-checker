from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import os


class LogisticClassifier:
    def __init__(self, regularization='l2', epochs=5):
        self._logistic_classifier = LogisticRegression(penalty=regularization, max_iter=epochs)

    def train(self, train_x, train_y):
        self._logistic_classifier.fit(train_x, train_y)

    def predict(self, test_x):
        return self._logistic_classifier.predict(test_x)

    def accuracy_test(self, test_x, test_y):
        accuracy = self._logistic_classifier.score(test_x, test_y)
        return accuracy

    def get_average_f1_score(self, x_test, y_test):
        labels = [1, 0, -1]
        y_pred = self._logistic_classifier.predict(x_test)

        # Save predicted labels
        project_relative_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        print(project_relative_path)
        output_file_sentiment_label = open(
            os.path.join(project_relative_path, 'saved_model_data/max_ent_labels.txt'), 'a')
        for label in y_pred:
            output_file_sentiment_label.write(str(label))
            output_file_sentiment_label.write('\n')

        return f1_score(y_test, y_pred, average='weighted', labels=labels)
