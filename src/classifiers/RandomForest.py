from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import os


class RandomForest:
    def __init__(self, n_estimators=50, max_depth=4):
        self._random_forest_classifier = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

    def train(self, train_x, train_y):
        self._random_forest_classifier.fit(train_x, train_y)

    def predict(self, test_x):
        return self._random_forest_classifier.predict(test_x)

    def accuracy_test(self, test_x, test_y):
        accuracy = self._random_forest_classifier.score(test_x, test_y)
        return accuracy

    def get_average_f1_score(self, x_test, y_test):
        labels = [1, 0, -1]
        y_pred = self._random_forest_classifier.predict(x_test)

        # Save predicted labels
        project_relative_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        print(project_relative_path)
        output_file_sentiment_label = open(
            os.path.join(project_relative_path, 'saved_model_data/random_forest_labels.txt'), 'a')
        for label in y_pred:
            output_file_sentiment_label.write(str(label))
            output_file_sentiment_label.write('\n')

        return f1_score(y_test, y_pred, average='weighted', labels=labels)
