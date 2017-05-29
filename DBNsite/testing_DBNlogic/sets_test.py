import pytest
import os
import numpy as np

import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..'))

from DBNlogic.sets import exists, full, DataSet, MNIST, SmallerMNIST

def test_existsTrue():
    """The function `exists` returns True if
    given the path of an existing file."""
    file = open('test.test', 'w+')
    file.close()
    assert exists('test.test')
    os.remove('test.test')

def test_existsFalse():
    """The function `exists` returns False if
    given the path of an unexisting file."""
    assert exists('test.test') == False

def test_DataSetConstructor():
    """A DataSet object can be constructed from an array."""
    data = [1, 2, 3, 4, 5, 6, 7]
    dataset = DataSet(data)
    assert all(dataset.data[i] == data[i] for i in range(len(data)))

def test_fromCSV():
    """A DataSet object can be constructed from a CSV file."""
    dataset = DataSet.fromCSV(full('left_8.csv'))
    assert dataset.data.shape == (14, 8)

def test_fromPickle():
    """A DataSet object can be constructed from a Pickle file."""
    dataset = DataSet.fromPickle(full('small_MNIST.pkl'))
    assert dataset.data.shape == (60000, 49)

def test_fromWhatever():
    """A DataSet object can be constructed from a file."""
    dataset = DataSet.fromWhatever('small_MNIST')
    assert dataset.data.shape == (60000, 49)

def test_allSets():
    """The available datasets include the MNIST dataset
    and a downsampling the MNIST dataset."""
    datasets = DataSet.allSets()
    assert 'MNIST' in datasets
    assert 'small_MNIST' in datasets

def test_MNIST():
    """A MNIST dataset object can be constructed from a file."""
    mnist = MNIST()
    assert mnist.data.shape == (60000, 784)

def test_SmallerMNIST():
    """A downsampling of the MNIST dataset
    can be constructed from a file."""
    mnist = SmallerMNIST()
    assert mnist.data.shape == (60000, 49)
