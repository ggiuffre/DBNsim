import numpy as np
import cudamat as cm



def startProcessor():
    """Start the processing unit."""
    cm.cublas_init()

def shutdownProcessor():
    """Shut the processing unit down."""
    cm.shutdown()

def matrix(A):
    """Return a representation of `A` suitable
    to the processing unit."""
    if type(A) == cm.CUDAMatrix or type(A) == cm.TransposedCUDAMatrix:
        return A
    elif type(A) == np.ndarray:
        return cm.CUDAMatrix(A)
    else:
        return cm.CUDAMatrix(np.array(A))

def asnumpy(A):
    """Return a NumPy copy of `A`."""
    if type(A) == np.ndarray:
        return A
    elif type(A) == cm.CUDAMatrix:
        return A.asarray()
    else:
        return np.array(A)

def transpose(A):
    """Return the transpose of `A`."""
    return matrix(A).transpose()

def dot(A, B):
    """Return the result of multiplying `A` with `B`."""
    return cm.dot(matrix(A), matrix(B))

def mul(c, A):
    """Return the result of multiplying `A`
    with the scalar `c`."""
    return matrix(A).mult(c)

def div(A, c):
    """Return the result of dividing `A` by the scalar `c`."""
    return matrix(A).divide(c)

def add(A, B):
    """Return the result of adding `A` with `B`."""
    return matrix(A).add(matrix(B))

def sub(A, B):
    """Return the result of subtracting `B` from `A`."""
    return matrix(A).subtract(matrix(B))

def cumsum(A, axis = None):
    """Return the cumulative sum of `A`."""
    return cm.sum(matrix(A), axis)

def sigmoid(A):
    """Return the element-wise sigmoid of `A`."""
    return cm.sigmoid(matrix(A))

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

def activation(A):
    """Return the element-wise binary activation of `A`."""
    rand_mat = cm.CUDAMatrix(np.random.uniform(size = A.shape))
    ones_mat = cm.CUDAMatrix(np.ones((A.shape))) # (-1, 1) |--> (0, 2)
    twos_mat = cm.CUDAMatrix(2 * np.ones((A.shape))) # (0, 2) |--> (0, 1)
    return matrix(A).subtract(rand_mat).sign().add(ones_mat).divide(twos_mat)

def squared_error(A, B):
    """Return the mean error between `A` and `B`."""
    return cm.sqrt(cm.pow(sub(A, B), 2).mean(axis = 0).mean(axis = 1)).asarray()[0, 0]
