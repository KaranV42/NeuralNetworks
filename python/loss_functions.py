import numpy as np

def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    res = -np.sum(y_true * np.log(y_pred + 1e-9)) / n_samples
    return res

def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true
