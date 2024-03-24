import numpy as np

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Loss_CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dvalues = dvalues.copy()
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues /= samples

# Your data and network setup
X = np.array([[6, 6, 4, 4, 6, 4, 3, 8, 8, 1, 9, 9, 8, 5, 6, 5, 7, 8, 6, 9, 5, 4, 2, 7, 7, 7, 7, 2, 5, 7, 3, 3, 2, 8, 8, 7, 3, 6, 6, 9], [4, 6, 7, 6, 6, 8, 1, 6, 4, 6, 5, 6, 7, 8, 7, 8, 9, 6, 2, 11, 7, 2, 8, 12, 11, 5, 9, 9, 8, 8, 3, 11, 3, 7, 6, 6, 6, 9, 4, 10], [6, 7, 8, 6, 7, 5, 1, 4, 5, 9, 5, 9, 2, 2, 8, 6, 5, 4, 2, 9, 6, 5, 13, 5, 7, 3, 12, 4, 5, 3, 7, 7, 4, 8, 5, 4, 3, 6, 2, 8], [9, 4, 2, 6, 8, 5, 1, 7, 8, 9, 8, 2, 3, 10, 7, 9, 6, 6, 10, 6, 2, 5, 4, 7, 8, 9, 2, 8, 4, 4, 9, 8, 7, 8, 3, 7, 2, 8, 4, 2], [4, 6, 10, 4, 7, 4, 8, 11, 6, 4, 1, 6, 10, 2, 3, 8, 2, 6, 5, 8, 7, 5, 6, 2, 5, 7, 7, 6, 6, 5, 4, 3, 3, 2, 4, 4, 6, 6, 7, 8], [5, 4, 6, 4, 5, 5, 3, 6, 8, 7, 7, 11, 2, 3, 8, 2, 7, 8, 10, 4, 5, 2, 11, 7, 2, 6, 3, 6, 5, 8, 3, 3, 4, 2, 7, 8, 4, 10, 8, 2], [6, 6, 4, 7, 7, 5, 6, 4, 6, 6, 8, 5, 6, 6, 1, 5, 9, 6, 8, 6, 6, 4, 2, 9, 5, 4, 2, 3, 5, 8, 7, 3, 8, 11, 3, 6, 2, 8, 5, 6], [7, 6, 4, 9, 7, 10, 4, 9, 7, 9, 9, 1, 4, 6, 9, 4, 2, 2, 11, 9, 8, 13, 4, 9, 4, 11, 10, 5, 7, 3, 7, 4, 7, 3, 3, 2, 6, 10, 4, 9], [9, 4, 7, 6, 6, 4, 7, 6, 8, 6, 4, 7, 9, 8, 9, 7, 8, 6, 7, 9, 5, 4, 2, 2, 8, 8, 4, 8, 5, 8, 3, 2, 4, 10, 7, 9, 7, 8, 7, 7], [4, 4, 4, 7, 6, 6, 6, 7, 9, 5, 9, 5, 2, 6, 6, 7, 6, 2, 6, 3, 7, 8, 6, 7, 8, 6, 6, 7, 9, 5, 3, 3, 2, 4, 8, 8, 7, 13, 12, 6], [4, 7, 9, 6, 9, 1, 10, 5, 4, 8, 8, 5, 6, 8, 2, 6, 12, 8, 7, 8, 7, 2, 7, 9, 6, 7, 8, 9, 5, 10, 7, 5, 7, 3, 3, 6, 2, 4, 5, 7]])
y = np.array([5, 8, 5, 10, 6, 7, 3, 10, 8, 4, 7])

dense1 = Layer_Dense(40, 10)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(10, 5)
activation2 = Activation_Softmax()

optimizer = Optimizer_SGD(learning_rate=0.1)

# Forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Calculate loss
loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.forward(activation2.output, y)

# Backward pass
loss_function.backward(loss, y)
activation2.backward(loss_function.dvalues)
dense2.backward(activation2.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Update weights and biases
optimizer.update_params(dense1)
optimizer.update_params(dense2)




