import numpy as np
from nets import DBN, RBM
from sets import SmallerMNIST
from plot import WeightsPlotter



L1  = RBM(49, 10)
net = DBN([L1], name = 'prova')

trainset = SmallerMNIST()
net.learn(trainset.data, max_epochs = 1)
net.save()

p = WeightsPlotter(net[0], (7, 7))
p.plot(2)
