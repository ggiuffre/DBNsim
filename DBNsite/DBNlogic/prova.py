import numpy as np
from nets import DBN, RBM
from sets import *
from util import plotImage
from train import CDTrainer
import matplotlib.pyplot as plt



L1  = RBM(49, 60)
net = DBN([L1], name = 'mnist')

# trainset = DataSet.fromPickle('data/left_8.pkl')
# trainset = DataSet.fromPickle('data/faces_all.pkl')
trainset = SmallerMNIST().data

net.learn(trainset, max_epochs = 10, batch_sz = 10)
net.save()
# net = DBN.load('mnist')
# t = ErrorPlotter(CDTrainer(net[0], max_epochs = 20), trainset)
# t.plot()

# for x in range(4):
#     print(net.evaluate(np.random.rand(8)).T > 0.5)

# plt.imshow(net.evaluate(np.random.rand(49)).reshape(7, 7))
# plt.show()

plotImage(net.evaluate(trainset[2]).reshape(7, 7))
