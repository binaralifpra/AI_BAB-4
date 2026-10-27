#Example 4.18 SimpleRNN.py
#https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
import tensorflow as tf
import numpy as np

i = 4   # batch size (and number of RNN units)
s = 5   # time steps
n = 1   # features per time step
inputs = np.random.random([i, s, n]).astype(np.float32)
simple_rnn = tf.keras.layers.SimpleRNN(i)
print(inputs)

output = simple_rnn(inputs)  # The output has shape `[4, 4]`.
print(output)
