## Neural Network Structure

This project implements a basic feedforward neural network from scratch using only NumPy. The network is designed to solve simple classification problems, such as the XOR problem, and consists of the following components:

### Architecture

- **Input Layer**: 
  - Size: `2` (for two input features, e.g., XOR input `[0, 0]`)
  
- **Hidden Layer**:
  - Size: `4` neurons
  - Activation Function: `ReLU` (Rectified Linear Unit)
  
- **Output Layer**:
  - Size: `2` neurons (for binary classification with one-hot encoded outputs)
  - Activation Function: `Softmax` (for multi-class classification)

### Forward Pass
1. **Input to Hidden Layer**:
   - The input data is multiplied by the weights and added to the bias for the hidden layer.
   - `Z1 = X . W1 + b1`
   - The result is passed through the `ReLU` activation function.
   - `A1 = ReLU(Z1)`
   
2. **Hidden Layer to Output Layer**:
   - The output from the hidden layer is multiplied by the second set of weights and added to the output bias.
   - `Z2 = A1 . W2 + b2`
   - The result is passed through the `Softmax` activation function.
   - `A2 = Softmax(Z2)`

### Backpropagation
1. **Loss Function**: 
   - The loss is calculated using `Categorical Cross-Entropy`, which measures the difference between the true labels and predicted probabilities.
   - `Loss = -Σ(y_true * log(y_pred)) / n_samples`
   
2. **Gradient Descent**:
   - The gradients of the loss function are computed with respect to each weight and bias using backpropagation.
   - The weights and biases are updated using the learning rate to minimize the loss function.

### Training
- The neural network is trained using gradient descent for a fixed number of epochs. After each forward pass, the network computes the loss and performs backpropagation to adjust the weights and biases.
- The model updates its parameters over time to reduce the error between the predicted output and the true labels.

### Example Usage

The network is trained on a simple XOR dataset with one-hot encoded labels:

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoding
