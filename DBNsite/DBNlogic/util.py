import numpy as np
from math import sqrt



class Configuration:
    """Hyper-parameters configuration for training a neural network."""

    def __init__(self,
        max_epochs = 10,    # - - - -
        threshold  = 0.05,  #
        batch_size = 1,     # default
        learn_rate = 0.1,   # values
        momentum   = 0.5,   #
        w_decay    = 0.0002 # - - - -
    ):
        """Construct a Configuration object from the
        given training hyper-parameters."""
        self.max_epochs = max_epochs # max n. of training epochs
        self.threshold  = threshold  # target error threshold
        self.batch_size = batch_size # size of a mini-batch
        self.learn_rate = learn_rate # learning rate
        self.momentum   = momentum   # learning momentum
        self.w_decay    = w_decay    # weight decay factor



def startProcessor():
    """Start the processing unit."""
    pass

def shutdownProcessor():
    """Shut the processing unit down."""
    pass

def matrix(A):
    """Return a representation of `A` suitable
    to the processing unit."""
    return np.array(A)

def asnumpy(A):
    """Return a NumPy copy of `A`."""
    if type(A) == np.ndarray:
        return A
    else:
        return np.array(A)

def transpose(A):
    """Return the transpose of `A`."""
    return A.T

def dot(A, B):
    """Return the result of multiplying `A` with `B`."""
    return np.dot(A, B)

def mul(c, A):
    """Return the result of multiplying `A`
    with the scalar `c`."""
    return A * c

def div(A, c):
    """Return the result of dividing `A` by the scalar `c`."""
    return A / c

def add(A, B):
    """Return the result of adding `A` with `B`."""
    return A + B

def sub(A, B):
    """Return the result of subtracting `B` from `A`."""
    return A - B

def cumsum(A, axis = None):
    """Return the cumulative sum of `A`."""
    return A.sum(axis = axis, keepdims = True)

def sigmoid(A):
    """Return the element-wise sigmoid of `A`."""
    ones = np.ones(A.shape) # (array of ones)
    return ones / (ones + np.exp(-A))

def repeat(A, times, axis):
    """Return `A` juxtaposed to itself `times` times,
    along the `axis` axis."""
    return np.repeat(A, times, axis)

def activation(A):
    """Return the element-wise binary activation of `A`."""
    return A > np.random.uniform(size = A.shape)

def squared_error(A, B):
    """Return the mean error between `A` and `B`."""
    return np.sqrt(((A - B) ** 2).mean())



def heatmap(array):
    """Return a Highcharts-formatted heatmap from a Python array."""
    dim = int(sqrt(len(array)))
    for row in range(dim):
        for col in range(dim):
            array[row * dim + col] = [col, dim - 1 - row, array[row * dim + col]] # from Python array to X,Y coordinates...
    return array
