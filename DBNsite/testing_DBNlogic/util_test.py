import pytest
import numpy as np
import math

import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..'))

from DBNlogic.util import sigmoid, activation, heatmap



def test_sigmoidDefinition():
    """The `sigmoid` function applied to `x` returns 1 / (1 + e^-x)."""
    for i in range(-1000,1001):
        x = 0.1 * i
        assert sigmoid(np.array(x)) == 1 / (1 + math.exp(-x))

def test_sigmoidShape():
    """The `sigmoid` function returns a Numpy array with
    the same shape as the input array."""
    v = np.random.randn(5, 12, 38)
    assert sigmoid(v).shape == v.shape

def test_sigmoidZero():
    """The `sigmoid` function applied to 0 returns 0.5."""
    x = 0
    assert sigmoid(np.array(x)) == 0.5

def test_sigmoidAsymptotes():
    """The `sigmoid` function always stays between 0 and 1."""
    for i in range(-100,101):
        s = sigmoid(np.array(0.23 * i))
        assert s > 0
        assert s < 1

def test_activationShape():
    """The `activation` function returns a Numpy array with
    the same shape as the input array."""
    v = np.random.randn(5, 12)
    assert activation(v).shape == v.shape

def test_heatmap():
    """A heatmap of an array has the same length as the array."""
    array = np.random.rand(16).tolist()
    hmap = heatmap(array)
    assert len(array) == len(hmap)
