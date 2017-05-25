import pytest
import os

import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..'))

import DBNlogic.nets as nets
from DBNlogic.nets import DBN, RBM

def test_full():
    """Given the base name of a network file,
    full() returns the full path to it."""
    net = DBN(name = 'Test')
    net.save()
    netFile = nets.full('Test.pkl')
    assert os.path.isfile(netFile)
    os.remove(netFile)

def test_append():
    """A RBM can be appended to a DBN."""
    net = DBN();
    net.append(RBM(3, 4))
    net.append(RBM(4, 7))
    assert len(net) == 2

def test_init():
    """A DBN can be constructed."""
    net = DBN([RBM(4, 7), RBM(7, 3)], 'Test')
    assert len(net) == 2

def test_generation():
    """A DBN can generate a sample whose length
    is equal to the number of visible units of
    its first RBM."""
    net = DBN([RBM(4, 7), RBM(7, 3)])
    sample = net.evaluate([0, 1, 1, 0.4])
    assert len(sample) == 4
