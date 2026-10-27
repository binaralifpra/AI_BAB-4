#Exercise 4.7
# Modify Example 4.9: change the number of filters in Conv2D from 32 to 64,
# then change the kernel size from (5,5) to (3,3), and compare performance.
import time
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

EPOCHS = 10   # lower this (e.g. 3) for a quicker comparison

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

def run(filters, kernel_size):
    keras.utils.set_random_seed(7)
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    start = time.time()
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, batch_size=200, verbose=2)
    train_time = time.time() - start
    scores = model.evaluate(X_test, y_test, verbose=0)
    return model.count_params(), train_time, 100 - scores[1] * 100

configs = [(32, (5, 5)), (64, (5, 5)), (32, (3, 3))]
results = []
for filters, kernel in configs:
    print("\n=== Conv2D(%d, %s) ===" % (filters, kernel))
    results.append(run(filters, kernel))

print("\n%-18s %-12s %-12s %-10s" % ("Conv2D", "Params", "Time (s)", "Error %"))
for (filters, kernel), (params, t, err) in zip(configs, results):
    print("%-18s %-12s %-12.1f %-10.2f" % ("%d, %s" % (filters, kernel),
                                           format(params, ","), t, err))
