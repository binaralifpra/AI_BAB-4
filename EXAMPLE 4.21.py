#Example 4.21 SimpleLSTM.py
#Based on
#https://www.tensorflow.org/guide/keras/rnn
import keras
from keras import layers

model = keras.Sequential()
# Keras 3 only builds the layers (and counts parameters) once it knows
# the input shape: a sequence of any length of integer word indices.
model.add(keras.Input(shape=(None,)))
model.add(layers.Embedding(input_dim=100, output_dim=32))
model.add(layers.LSTM(64))
model.add(layers.Dense(1))
model.summary()
