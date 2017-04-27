from __future__ import print_function
import numpy as np
import math
import random



def sigmoid(a):
    """Return the sigmoid of a vector."""
    return [1.0 / (1.0 + math.exp(-x)) for x in a]

def activation(a):
    """Return a binary activation vector."""
    return [x > random.uniform(0, 1) for x in sigmoid(a)]

def correlation(a, b):
    """Return the binary correlation between 2 vectors."""
    return [[i * j for j in b] for i in a]



class CDTrainer:
    """Contrastive Divergence trainer for RBMs."""

    def __init__(self, net, threshold = 0.05, max_epochs = 100):
        """Constructor for a RBM."""
        self.net = net
        self.threshold = threshold # target error
        self.max_epochs = 100 # max n. of epochs for training
        self.eta = 0.5 # learning rate

    def __str__(self):
        """Return the weights representing the RBM."""
        return 'Contrastive Divergence trainer'

    def __repr__(self):
        """Return the dimensions of the RBM."""
        return 'CDTrainer(' + repr(self.net) + ')'

    def run(self, trainset):
        """Learn from a particular training set."""
        net = self.net
        print('-----------')
        print('training...')
        done  = False
        epoch = 0
        while (not done) and (epoch < self.max_epochs):
            epoch += 1
            for example in trainset:
                # positive phase:
                net.set(example)
                pos_prods   = np.array(correlation(net.v, net.h))
                pos_vis_act = np.array(net.v)
                pos_hid_act = np.array(net.h)

                # negative phase:
                net.get()
                neg_prods   = np.array(correlation(net.v, net.h))
                neg_vis_act = np.array(net.v)
                neg_hid_act = activation(np.dot(net.v, net.W) + net.b)

                # update:
                net.W += net.eta * (pos_prods - neg_prods)
                net.a += net.eta * (pos_vis_act - neg_vis_act)
                net.b += net.eta * (pos_hid_act - neg_hid_act)

            # error calculation:
            done = all([all([abs(x) < self.threshold for x in row]) for row in pos_prods - neg_prods])

        print('done after', epoch, 'epochs.')
        print('-----------')
