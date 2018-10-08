from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import os


class LinearSupportVectorClassifier:
    def __init__(self, regularization='l2', loss='squared_hinge', C=1.0, max_iter=1000):
        self._linear_svc = LinearSVC(penalty=regularization, loss=loss, C=C, max_iter=max_iter)

    def train(self, train_x, train_y):
        self._linear_svc.fit(train_x, train_y)

    def predict(self, test_x):
        return self._linear_svc.predict(test_x)

    def accuracy_test(self, test_x, test_y):
        accuracy = self._linear_svc.score(test_x, test_y)
        return accuracy

    def get_average_f1_score(self, x_test, y_test):
        labels = [1, 0, -1]
        y_pred = self._linear_svc.predict(x_test)

        # Save predicted labels
        project_relative_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        print(project_relative_path)
        output_file_sentiment_label = open(
            os.path.join(project_relative_path, 'saved_model_data/svm_labels.txt'), 'a')
        for label in y_pred:
            output_file_sentiment_label.write(str(label))
            output_file_sentiment_label.write('\n')

        return f1_score(y_test, y_pred, average='weighted', labels=labels)