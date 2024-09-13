import numpy as np
from neural_network import NeuralNetwork

if __name__ == "__main__":
    # Example dataset (XOR problem with one-hot encoded labels)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoding

    # Create the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=2)

    # Train the network
    nn.train(X, y, epochs=1000, learning_rate=0.1)

    # Test the network
    predictions = nn.forward(X)
    print("Predictions:\n", np.round(predictions, 2))