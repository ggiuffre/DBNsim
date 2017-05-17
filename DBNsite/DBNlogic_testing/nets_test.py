import pytest

import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath, '..'))

from DBNlogic.nets import DBN, RBM

def test_deepNet():
    net = DBN();
    net.append(RBM(3, 4))
