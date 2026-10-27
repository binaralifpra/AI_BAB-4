#Exercise 4.6
# Modify Example 4.7 and add another Conv2D, MaxPooling and Dropout layer
# before the Flatten layer. Compare the number of parameters.
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def build(extra_block):
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    if extra_block:
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

original = build(extra_block=False)
modified = build(extra_block=True)
modified.summary()

print("Original model params:", format(original.count_params(), ","))
print("Modified model params:", format(modified.count_params(), ","))
# The extra block adds a few conv weights but shrinks the Flatten output
# (12*12*32 = 4608 -> 5*5*64 = 1600), so the Dense layer and the total
# parameter count get much smaller.
