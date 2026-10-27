#Example 4.1 Neuron.py
# A single neuron (perceptron) that learns the logical AND of two inputs.
# Note: the book's listing uses outputs [0, 1, 1, 1], which is actually OR,
# and has no bias term. Without a bias a single neuron cannot learn AND
# (input [0, 0] always gives 0.5), so a constant bias input of 1 is added
# as the third column.
from numpy import exp, array, random, dot

#Set up parameters ===============================================
#            x1 x2 bias
inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
outputs = array([[0, 1, 0, 0]]).T          # AND
random.seed(1)
weights = 2 * random.random((3, 1)) - 1

#Define a Single Neuron function =================================
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output

#Train the Neuron ================================================
for iteration in range(50000):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

#Test the Neuron =================================================
for x in [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]:
    print(x[:2], "->", neuron(array(x), weights))
