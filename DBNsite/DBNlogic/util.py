import numpy as np



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



def sigmoid(v):
    """Return the element-wise sigmoid of a Numpy array."""
    ones = np.ones(v.shape) # (matrix of ones)
    return ones / (ones + np.exp(-v))

def activation(v):
    """Return the element-wise binary activation of a Numpy array."""
    return v > np.random.uniform(size = v.shape)

def squared_error(v, w):
    """Return the mean squared error between two Numpy arrays."""
    return ((v - w) ** 2).mean() # TODO: add sqrt?
