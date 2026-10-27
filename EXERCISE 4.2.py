#Exercise 4.2
# Modify Example 4.2 so that it has three inputs instead of two.
# The network learns the logical AND of three inputs (all 8 combinations).
from sklearn.neural_network import MLPClassifier

X = [[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.],
     [1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]]
y = [0, 0, 0, 0, 0, 0, 0, 1]               # x1 AND x2 AND x3

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

print(clf.predict([[2., 2., 2.], [-1., -2., 1.]]))
print(clf.predict(X))
# The first weight matrix is now (3, 5): 3 inputs into 5 hidden neurons
print([coef.shape for coef in clf.coefs_])
