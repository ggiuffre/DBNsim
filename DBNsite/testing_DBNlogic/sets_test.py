import pytest
import os
import numpy as np

import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..'))

from DBNlogic.sets import exists, full, DataSet, SCIPY_AVAILABLE



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

def test_save():
    """A DataSet object can be saved to a CSV, Pickle, or Matlab file."""
    a = DataSet(np.random.rand(8, 5))
    a.save('test.mat')
    if SCIPY_AVAILABLE:
        assert exists('test.mat')
        os.remove('test.mat')

def test_fromCSV():
    """A DataSet object can be constructed from a CSV file."""
    dataset = DataSet.fromCSV(full('top_left.csv'))
    assert dataset.shape == (10, 16)

def test_fromPickle():
    """A DataSet object can be constructed from a Pickle file."""
    dataset = DataSet.fromPickle(full('top_left.pkl'))
    assert dataset.shape == (10, 16)

def test_fromMatlab():
    """A DataSet object can be constructed from a Matlab file."""
    a = np.random.rand(8, 5)
    DataSet(a).save('test.mat')
    b = DataSet.fromMatlab('test.mat')
    if SCIPY_AVAILABLE:
        assert a.shape == b.shape
        os.remove('test.mat')
    else:
        assert len(b) == 0

def test_fromWhatever():
    """A DataSet object can be constructed from a file."""
    dataset = DataSet.fromWhatever('top_left')
    assert dataset.shape == (10, 16)
