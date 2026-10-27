#Exercise 4.5
# Modify the AlexNet program from Example 4.5 to use different numbers of
# dense layer units and see how this affects the total number of parameters.
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def alexnet(fc1, fc2, fc3):
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (11, 11), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(fc1, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(fc2, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(fc3, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(17, activation='softmax'))
    return model

print("%-20s %-14s" % ("Dense units", "Total params"))
for units in [(4096, 4096, 1000), (2048, 2048, 1000), (1024, 1024, 512), (512, 256, 128)]:
    print("%-20s %-14s" % (units, format(alexnet(*units).count_params(), ",")))

# The conv layers stay at ~6.1M params; the dense layers dominate, and
# Dense(fc1)->Dense(fc2) grows with fc1*fc2, so halving both units cuts
# that layer to about a quarter.
