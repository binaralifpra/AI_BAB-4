#Exercise 4.4
# Modify the LeNet program from Example 4.4 to use different numbers of
# filters and see how this affects the total number of parameters.
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, AveragePooling2D

def lenet(filters1, filters2):
    model = Sequential()
    model.add(Input(shape=(32, 32, 1)))
    model.add(Conv2D(filters=filters1, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=filters2, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    return model

print("%-10s %-10s %-12s" % ("Conv1", "Conv2", "Total params"))
for f1, f2 in [(6, 16), (3, 8), (12, 32), (32, 64)]:
    print("%-10d %-10d %-12s" % (f1, f2, format(lenet(f1, f2).count_params(), ",")))

# Most parameters sit in the first Dense layer: Flatten gives 6*6*filters2
# values, so doubling filters2 roughly doubles that layer's weights.
lenet(12, 32).summary()
