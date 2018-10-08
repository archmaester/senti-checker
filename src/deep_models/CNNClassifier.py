import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


class CNNClassifier:
    def __init__(self, input_shape, num_classes):
        self._model = Sequential()
        self._model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self._model.add(Conv2D(32, (3, 3), activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))
        self._model.add(Dropout(0.25))

        self._model.add(Conv2D(64, (3, 3), activation='relu'))
        self._model.add(Conv2D(64, (3, 3), activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))
        self._model.add(Dropout(0.25))

        self._model.add(Flatten())
        self._model.add(Dense(256, activation='relu'))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(num_classes, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self._model.compile(loss='categorical_crossentropy', optimizer=sgd)

    def train(self, train_x, train_y):
        print(train_x.shape)
        assert len(train_y) == train_x.shape[0]
        train_y = np.reshape(train_y, (len(train_y), 1))
        self._model.fit(train_x, train_y, epochs=10, batch_size=100)

    def accuracy_test(self, test_x, test_y):
        assert (len(test_y) == test_x.shape[0])
        test_y = np.reshape(test_y, (len(test_y), 1))
        accuracy = self._model.evaluate(test_x, test_y, batch_size=16)[1]
        return accuracy