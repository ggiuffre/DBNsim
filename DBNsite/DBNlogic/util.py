import numpy as np
from math import sqrt



class Configuration:
    """Hyper-parameters configuration for training a neural network."""

    def __init__(self,
        max_epochs = 10,     # -  -  -
        threshold  = 0.05,   #
        batch_size = 1,      #
        learn_rate = 0.1,    # default
        momentum   = 0.9,    # values
        w_decay    = 0.0002, #
        sparsity   = 0.2,    #
        std_dev    = 0.01    # -  -  -
    ):
        """Construct a Configuration object from the
        given training hyper-parameters."""
        self.max_epochs = max_epochs # max n. of training epochs
        self.threshold  = threshold  # target error threshold
        self.batch_size = batch_size # size of a mini-batch
        self.learn_rate = learn_rate # learning rate
        self.momentum   = momentum   # learning momentum
        self.w_decay    = w_decay    # weight decay factor
        self.sparsity   = sparsity   # sparsity index
        self.std_dev    = std_dev    # std. dev. of the weights distribution



def sigmoid(A):
    """Return the element-wise sigmoid of `A`."""
    ones = np.ones(A.shape) # (array of ones)
    return ones / (ones + np.exp(-A))

def activation(A):
    """Return the element-wise binary activation of `A`."""
    return A > np.random.uniform(size = A.shape)



def heatmap(array):
    """Return a Highcharts-formatted heatmap from a Python list."""
    dim = int(sqrt(len(array)))
    for row in range(dim):
        for col in range(dim):
            array[row * dim + col] = [col, dim - 1 - row, array[row * dim + col]] # from Python array to X,Y coordinates...
    return array
