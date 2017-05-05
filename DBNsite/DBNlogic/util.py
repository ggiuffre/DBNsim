import numpy as np
import matplotlib.pyplot as plt

def sigmoid(v):
    """Return the element-wise sigmoid of a vector or matrix."""
    ones = np.ones(v.shape) # matrix of ones
    if v.any() > 1:
        print('high value!')
    return ones / (ones + np.exp(-v))

def activation(v):
    """Return the element-wise binary activation of a vector or matrix."""
    return v > np.random.uniform(size = v.shape)

def squared_error(v, w):
    """Return the mean squared error between two vectors or matrices."""
    return ((v - w) ** 2).mean() # TODO: add sqrt!

def plotImage(img, shape = None):
    if shape != None:
        plt.imshow(img.reshape(shape))
    else:
        plt.imshow(img)
    plt.colorbar()
    plt.show()
