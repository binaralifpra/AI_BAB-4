#Exercise 4.1
# Modify Example 4.1 so that the neuron performs a logical OR of the inputs.
# Only the target outputs change: OR is 1 whenever at least one input is 1.
from numpy import exp, array, random, dot

#Set up parameters ===============================================
#            x1 x2 bias
inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
outputs = array([[0, 1, 1, 1]]).T          # OR
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
