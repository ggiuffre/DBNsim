import pytest

import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..'))

from DBNlogic.nets import DBN, RBM

def test_append():
    net = DBN();
    net.append(RBM(3, 4))
    net.append(RBM(4, 7))
    assert len(net) == 2

def test_init():
    net = DBN([RBM(4, 7), RBM(7, 3)], 'Test')

def test_generation():
    net = DBN([RBM(4, 7), RBM(7, 3)])
    sample = net.evaluate([0, 1, 1, 0.4])
    assert len(sample) == 4
