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



class ProcessingUnit:
    """Matrix operations for a particular
    type of processing unit."""
    pass

class CPU(ProcessingUnit):
    """Matrix operations for a normal CPU."""

    @staticmethod
    def start():
        """Start the processing unit."""
        pass

    @staticmethod
    def shutdown():
        """Shut the processing unit down."""
        pass

    @staticmethod
    def matrix(A):
        """Return a representation of `A` suitable
        to the processing unit."""
        return np.array(A)

    @staticmethod
    def numpy(A):
        """Return a NumPy copy of `A`."""
        if type(A) == np.ndarray:
            return A
        else:
            return np.array(A)

    @staticmethod
    def transpose(A):
        """Return the transpose of `A`."""
        return A.T

    @staticmethod
    def dot(A, B):
        """Return the result of multiplying `A` with `B`."""
        return np.dot(A, B)

    @staticmethod
    def div(A, B):
        """Return the result of dividing `A` by `B`."""
        return A / B

    @staticmethod
    def mul(c, A):
        """Return the result of multiplying `A`
        with the scalar `c`."""
        return A * c

    @staticmethod
    def add(A, B):
        """Return the result of adding `A` with `B`."""
        return A + B

    @staticmethod
    def sub(A, B):
        """Return the result of subtracting `B` from `A`."""
        return A - B

    @staticmethod
    def cumsum(A, axis = None):
        """Return the cumulative sum of `A`."""
        return A.sum(axis = axis, keepdims = True)

    @staticmethod
    def sigmoid(A):
        """Return the element-wise sigmoid of `A`."""
        ones = np.ones(A.shape) # (array of ones)
        return ones / (ones + np.exp(-A))

    @staticmethod
    def repeat(A, times, axis):
        """Return `A` juxtaposed to itself `times` times,
        along the `axis` axis."""
        return np.repeat(A, times, axis)

    @staticmethod
    def activation(A):
        """Return the element-wise binary activation of `A`."""
        return A > np.random.uniform(size = A.shape)

class CUDA(ProcessingUnit):
    """Matrix operations for a CUDA-enabled GPU."""

    @staticmethod
    def start():
        """Start the processing unit."""
        cm.cublas_init()

    @staticmethod
    def shutdown():
        """Shut the processing unit down."""
        cm.shutdown()

    @staticmethod
    def matrix(A):
        """Return a representation of `A` suitable
        to the processing unit."""
        if type(A) == cm.CUDAMatrix:
            return A
        elif type(A) == np.ndarray:
            return cm.CUDAMatrix(A)
        else:
            return cm.CUDAMatrix(np.array(A))

    @staticmethod
    def numpy(A):
        """Return a NumPy copy of `A`."""
        if type(A) == np.ndarray:
            return A
        elif type(A) == cm.CUDAMatrix:
            return A.asarray()
        else:
            return np.array(A)

    @staticmethod
    def transpose(A):
        """Return the transpose of `A`."""
        return matrix(A).transpose()

    @staticmethod
    def dot(A, B):
        """Return the result of multiplying `A` with `B`."""
        return cm.dot(matrix(A), matrix(B))

    @staticmethod
    def div(A, B):
        """Return the result of dividing `A` by `B`."""
        return matrix(A).divide(B)

    @staticmethod
    def mul(c, A):
        """Return the result of multiplying `A`
        with the scalar `c`."""
        return matrix(A).mult(c)

    @staticmethod
    def add(A, B):
        """Return the result of adding `A` with `B`."""
        return matrix(A).add(B)

    @staticmethod
    def sub(A, B):
        """Return the result of subtracting `B` from `A`."""
        return matrix(A).subtract(B)

    @staticmethod
    def cumsum(A, axis = None):
        """Return the cumulative sum of `A`."""
        return cm.sum(matrix(A), axis)

    @staticmethod
    def sigmoid(A):
        """Return the element-wise sigmoid of `A`."""
        return cm.sigmoid(matrix(A))

    @staticmethod
    def repeat(A, times, axis):
        """Return `A` juxtaposed to itself `times` times,
        along the `axis` axis.
        N.B. Currently implemented only for juxtaposing
        vectors (vertical arrays)!"""
        if A.shape[1] == 1 and axis == 1:
            multiplier = cm.CUDAMatrix(np.ones((1, times)))
            return cm.dot(matrix(A), multiplier)
        else:
            raise NotImplementedError

    @staticmethod
    def activation(A):
        """Return the element-wise binary activation of `A`."""
        rand_mat = cm.CUDAMatrix(np.random.uniform(size = A.shape))
        ones_mat = cm.CUDAMatrix(np.ones((A.shape))) # (-1, 1) |--> (0, 2)
        twos_mat = cm.CUDAMatrix(2 * np.ones((A.shape))) # (0, 2) |--> (0, 1)
        return A.subtract(rand_mat).sign().add(ones_mat).divide(twos_mat)



def sigmoid(v):
    """Return the element-wise sigmoid of a Numpy array."""
    ones = np.ones(v.shape) # (array of ones)
    return ones / (ones + np.exp(-v))

def activation(v):
    """Return the element-wise binary activation of a Numpy array."""
    return v > np.random.uniform(size = v.shape)

def squared_error(v, w):
    """Return the mean squared error between two Numpy arrays."""
    return np.sqrt(((v - w) ** 2).mean())



def heatmap(array):
    """Return a Highcharts-formatted heatmap from a Python array."""
    dim = int(sqrt(len(array)))
    for row in range(dim):
        for col in range(dim):
            array[row * dim + col] = [col, dim - 1 - row, array[row * dim + col]] # from Python array to X,Y coordinates...
    return array
