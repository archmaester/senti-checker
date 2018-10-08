import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import os


class LSTMClassifier:
    def __init__(self, max_features=None):
        get_custom_objects().update({'swish': LSTMClassifier.__swish})
        self._model = Sequential()
        self._model.add(Embedding(max_features, output_dim=50))
        # self._model.add(LSTM(128, activation='relu', return_sequences=True))
        self._model.add(LSTM(64, activation='relu'))
        self._model.add(Dropout(0.35))
        self._model.add(Dense(3, activation='sigmoid'))

        self._model.compile(loss='binary_crossentropy',
                            optimizer='rmsprop',
                            metrics=['accuracy'])
        self._encoder = LabelEncoder()

    def train(self, train_x, train_y):
        print(train_x.shape)
        assert len(train_y) == train_x.shape[0]
        self._encoder.fit(train_y)
        train_y = self._encoder.transform(train_y)
        train_y = to_categorical(train_y)
        # train_y = np.reshape(train_y, (len(train_y), 1))
        self._model.fit(train_x, train_y, epochs=100, batch_size=256)

    def accuracy_test(self, test_x, test_y):
        assert (len(test_y) == len(test_x))
        test_y = to_categorical(self._encoder.transform(test_y))
        # test_y = np.reshape(test_y, (len(test_y), 1))
        accuracy = self._model.evaluate(test_x, test_y, batch_size=128)[1]
        return accuracy

    def get_average_f1_score(self, x_test, y_test):
        labels = [1, 0, -1]
        label_category = to_categorical(self._encoder.transform(labels))
        label_category_positive = [int(label) for label in label_category[0, :]]
        label_category_neutral = [int(label) for label in label_category[1, :]]
        label_category_negative = [int(label) for label in label_category[2, :]]

        print(label_category_positive)
        print(label_category_neutral)
        print(label_category_negative)

        y_pred = self._model.predict(x_test)
        y_pred_labels = []
        for i in range(y_pred.shape[0]):
            y_pred_i = y_pred[i, :]
            max_val = np.max(y_pred_i)
            y_pred_i = (y_pred_i >= max_val).astype(int)
            if np.array_equal(y_pred_i, label_category_positive):
                y_pred_labels.append('1')
            elif np.array_equal(y_pred_i, label_category_neutral):
                y_pred_labels.append('0')
            else:
                y_pred_labels.append('-1')

        # Save predicted labels
        project_relative_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        print(project_relative_path)
        output_file_sentiment_label = open(os.path.join(project_relative_path, 'saved_model_data/lstm_labels.txt'), 'a')
        for label in y_pred_labels:
            output_file_sentiment_label.write(str(label))
            output_file_sentiment_label.write('\n')

        return f1_score(y_test, y_pred_labels, average='weighted', labels=labels)

    @staticmethod
    def __swish(x):
        return K.sigmoid(x) * x