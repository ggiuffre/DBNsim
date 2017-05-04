import numpy as np
from nets import DBN, RBM
from sets import *
from plot import WeightsPlotter, ErrorPlotter
from train import CDTrainer
import matplotlib.pyplot as plt



L1  = RBM(8, 15)
net = DBN([L1], name = 'left')

trainset = DataSet.fromPickle('data/left_8.pkl')
# trainset = SmallerMNIST().data

net.learn(trainset, max_epochs = 10)
net.save()
# t = ErrorPlotter(CDTrainer(net[0]), trainset[:6])
# t.plot()

for x in range(4):
    print(net.evaluate(np.random.rand(8)).T > 0.5)

# plt.imshow(net.evaluate(np.random.rand(49)).reshape(7, 7))
# plt.show()

# p = WeightsPlotter(net[0], (8, 1))
# p.plot(0)
