import pytest
import os
import numpy as np

import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..'))

from DBNlogic.sets import exists, DataSet, MNIST, SmallerMNIST

def test_exists_true():
    """The function `exists` returns True if
    given the path of an existing file."""
    file = open('test.test', 'w+')
    file.close()
    assert exists('test.test')
    os.remove('test.test')

def test_exists_false():
    """The function `exists` returns False if
    given the path of an unexisting file."""
    assert exists('test.test') == False
