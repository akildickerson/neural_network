import numpy as np


class NeuralNetwork:
    def __init__(self, layers, X, Y):
        self.layers = layers
        self.X = X
        self.Y = Y
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            w = np.random.randn(
                layers[i + 1], layers[i]
            )  # inputs determine the shape of the output matrix [layers[i] x layers[i+1]]
            b = np.random.randn(
                layers[i + 1], 1
            )  # starts at layers[i+1] because input layer doesn't get biases: shape -> [layers[i+1] x 1]
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, arr):
        return 1 / (1 + np.exp(-1 * arr))

    def sigmoid_derivative(self, arr):
        sig = self.sigmoid(arr)
        return sig * (1 - sig)

    def forward_pass(self, X):
        activations = [X]
        z = []

        for w, b in zip(self.weights, self.biases):
            Z = w @ activations[-1] + b
            A = self.sigmoid(Z)
            z.append(Z)
            activations.append(A)

        y_hat = activations[-1]
        return y_hat, activations, z

    def cost(self, y_hat, Y):
        losses = -((Y * np.log(y_hat)) + (1 - Y) * np.log(1 - y_hat))
        m = y_hat.reshape(-1).shape[0]
        average_losses = (1 / m) * np.sum(losses, axis=1)
        return np.sum(average_losses)

    def back_propogation(self, y_hat, activations, z, learning_rate=0.01):
        m = self.Y.shape[1]
        dA = -(self.Y / y_hat) + ((1 - self.Y) / (1 - y_hat))  # dL/dA^L

        for l in reversed(range(len(self.layers) - 1)):
            A_prev = activations[l]
            Z = z[l]

            dZ = dA * self.sigmoid_derivative(Z)
            dW = (1 / m) * dZ @ A_prev.T
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.weights[l] -= dW * learning_rate
            self.biases[l] -= db * learning_rate

            dA = self.weights[l].T @ dZ

    def train(self, epochs=50000, learning_rate=0.1):
        costs = []

        for i in range(epochs):
            y_hat, activations, z = self.forward_pass(self.X)
            cost = self.cost(y_hat, self.Y)
            costs.append(cost)
            y_hat_labels = (y_hat > 0.5).astype(int)

            accuracy = np.mean(y_hat_labels == self.Y)
            self.back_propogation(y_hat, activations, z, learning_rate)
            if i % 500 == 0:
                print(f"Epoch {i}: cost = {cost:.4f}, accuracy = {accuracy * 100:.2f}%")
        return costs
