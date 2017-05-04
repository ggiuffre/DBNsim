import numpy as np

def sigmoid(v):
    """Return the element-wise sigmoid of a vector."""
    ones = np.ones(v.shape) # array of ones
    return ones / (ones + np.exp(-v))

def activation(v):
    """Return the element-wise binary activation of a vector."""
    return v > np.random.uniform(size = v.shape)

def squared_error(v, w):
    return ((v - w) ** 2).mean()
