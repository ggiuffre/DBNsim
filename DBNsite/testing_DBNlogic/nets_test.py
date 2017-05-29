import pytest
import os
import numpy as np

import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..'))

import DBNlogic.nets as nets
from DBNlogic.nets import DBN, RBM
from DBNlogic.sets import DataSet
from DBNlogic.util import squared_error



def test_append():
    """A RBM can be appended to a DBN."""
    net = DBN();
    net.append(RBM(3, 4))
    net.append(RBM(4, 7))
    assert len(net) == 2

def test_init():
    """A DBN can be constructed from a list of RBMs."""
    net = DBN([RBM(4, 7), RBM(7, 3)], 'Test')
    assert len(net) == 2

def test_generationLength():
    """A DBN can generate a sample whose length
    is equal to the number of visible units of
    its first RBM."""
    net = DBN([RBM(4, 7), RBM(7, 3)])
    sample = net.evaluate([0, 1, 1, 0.4])
    assert len(sample) == 4

def test_save():
    """A DBN can be serialised and saved to a Pickle file."""
    net = DBN(name = 'Test')
    net.save()
    netFile = nets.full('Test.pkl')
    assert os.path.isfile(netFile)
    os.remove(netFile)

def test_load():
    """A DBN can be loaded from a Pickle file."""
    net = DBN([RBM(4, 5), RBM(5, 6)], 'Test')
    net.save()
    net = DBN.load('Test')
    assert len(net) == 2
    os.remove(nets.full('Test.pkl'))

def test_saveAndLoad():
    """A DBN serialised to a file is equivalent to
    a DBN loaded from the same file."""
    net_1 = DBN([RBM(4, 5), RBM(5, 6)], 'Test')
    net_1.save()
    net_2 = DBN.load('Test')
    assert [np.equal(net_1[i].W, net_2[i].W) for i in range(2)]
    os.remove(nets.full('Test.pkl'))

def test_learn():
    """A DBN can learn to generate samples from a dataset."""
    net = DBN([RBM(8, 15), RBM(15, 10)], 'Test')
    trainset = DataSet.fromWhatever('left_8')
    net.learn(trainset)

def test_error():
    """The reconstruction error of a DBN is less than 1."""
    net = DBN([RBM(8, 15), RBM(15, 10)], 'Test')
    trainset = DataSet.fromWhatever('left_8')
    net.learn(trainset)
    mean_err = squared_error(trainset[0], net.evaluate(trainset[0]))
    assert mean_err <= 1
