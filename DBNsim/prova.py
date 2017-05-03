import numpy as np
from nets import DBN, RBM
from sets import DataSet
from plot import WeightsPlotter, ErrorPlotter
from train import CDTrainer



L1  = RBM(8, 15)
net = DBN([L1], name = 'left')

trainset = DataSet.fromPickle('data/left_8.pkl')

net.learn(trainset, max_epochs = 20)
# t = ErrorPlotter(CDTrainer(net[0]), trainset[:6])
# t.plot()

# for x in range(10):
#     print(net.evaluate(np.random.rand(8)))

# p = WeightsPlotter(net[0], (8, 1))
# p.plot(0)
