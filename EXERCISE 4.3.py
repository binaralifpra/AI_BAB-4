#Exercise 4.3
# Modify Example 4.3 to use five neurons in the dense layers instead of 10,
# and compare training time and accuracy.
import time
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(whole_data.data, whole_data.target,
                                                    test_size=0.3, random_state=7)
features = X_train.shape[1]

def build_and_train(units):
    keras.utils.set_random_seed(7)
    model = Sequential()
    model.add(Input(shape=(features,)))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    start = time.time()
    model.fit(X_train, y_train, batch_size=50, validation_split=0.2,
              epochs=100, verbose=0)
    train_time = time.time() - start
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    return model.count_params(), train_time, loss, acc

print("%-8s %-8s %-12s %-8s %-8s" % ("Units", "Params", "Time (s)", "Loss", "Accuracy"))
for units in [10, 5]:
    params, t, loss, acc = build_and_train(units)
    print("%-8d %-8d %-12.2f %-8.4f %-8.4f" % (units, params, t, loss, acc))
