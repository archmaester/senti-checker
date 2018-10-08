import numpy as np
import os
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


class MLPClassifier:
    def __init__(self, input_shape=None):
        self._num_classes = 3
        self._model = Sequential()
        self._model.add(Dense(32, input_dim=input_shape, activation='relu'))
        self._model.add(Dropout(0.3))
        self._model.add((Dense(32, activation='relu')))
        self._model.add(Dropout(0.3))
        self._model.add(Dense(self._num_classes, activation='softmax'))

        self._model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
        self._encoder = LabelEncoder()

    def train(self, train_x, train_y):
        assert len(train_y) == train_x.shape[0]
        self._encoder.fit(train_y)
        train_y = self._encoder.transform(train_y)
        train_y = to_categorical(train_y)
        # train_y = np.reshape(train_y, (len(train_y), 1))
        self._model.fit(train_x, train_y, epochs=20, batch_size=128)

    def accuracy_test(self, test_x, test_y):
        assert (len(test_y) == test_x.shape[0])
        test_y = to_categorical(self._encoder.transform(test_y))
        # test_y = np.reshape(test_y, (len(test_y), 1))
        accuracy = self._model.evaluate(test_x, test_y, batch_size=128)[1]
        return accuracy

    def _save_model(self):
        relative_path_project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if not os.path.exists(os.path.join(relative_path_project_dir, "saved_model_data/mlp")):
            os.makedirs(os.path.join(relative_path_project_dir, "output_data"))
        # self._model.save()

    def get_average_f1_score(self, x_test, y_test):
        labels = [1, 0, -1]
        label_category = to_categorical(self._encoder.transform(labels))
        y_pred = self._model.predict_on_batch(x_test)
        y_pred_labels = []
        for i in range(y_pred.shape[0]):
            y_pred_i = y_pred[i, :]
            max_val = np.max(y_pred_i)
            y_pred_i = (y_pred_i >= max_val).astype(int)
            if np.array_equal(y_pred_i, label_category[0, :]):
                y_pred_labels.append('1')
            elif np.array_equal(y_pred_i, label_category[1, :]):
                y_pred_labels.append('0')
            else:
                y_pred_labels.append('-1')

        # Save predicted labels
        project_relative_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        print(project_relative_path)
        output_file_sentiment_label = open(
            os.path.join(project_relative_path, 'saved_model_data/mlp_labels.txt'), 'a')
        for label in y_pred_labels:
            output_file_sentiment_label.write(str(label))
            output_file_sentiment_label.write('\n')

        return f1_score(y_test, y_pred_labels, average='weighted', labels=labels)