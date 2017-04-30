import numpy as np
import matplotlib.pyplot as plt
from nets import DBN, RBM
from plot import WeightsPlotter
from sets import MNIST



L1  = RBM(784, 24)
net = DBN([L1])

trainset = MNIST()
examples = np.array([ex['image'] for ex in trainset.data]).reshape(60000, 784)

net.learn(examples[:1], max_epochs = 1)

plt.imshow(net[0].W.reshape((len(net[0].h), 28, 28))[0])
plt.show()
