import numpy as np
from activation_functions import relu, relu_derivative, softmax
from loss_functions import cross_entropy_loss, cross_entropy_derivative

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y_true, y_pred, learning_rate):
        # Backpropagation
        m = X.shape[0]
        
        # Gradient of loss with respect to output
        dZ2 = cross_entropy_derivative(y_true, y_pred)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Backprop through hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y_true, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = cross_entropy_loss(y_true, y_pred)
            self.backward(X, y_true, y_pred, learning_rate)
            if (epoch+1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}')