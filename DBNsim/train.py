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
    """Return the correlation between two binary vectors."""
    return [[i * j for j in b] for i in a]

def squared_error(a, b):
    return ((a - b) ** 2).mean()



class CDTrainer:
    """Contrastive Divergence trainer for RBMs."""

    def __init__(self, net, threshold = 0.05, max_epochs = 100):
        """Constructor for a RBM."""
        self.net = net               # the learner
        self.mean_squared_err = None # current error
        self.threshold = threshold   # target error
        self.epoch = 0               # current number of epochs
        self.max_epochs = max_epochs # max n. of epochs for training
        self.learn_rate = 0.5        # learning rate

    def run(self, trainset):
        """Learn from a particular training set."""
        net = self.net
        print('-----------')
        print('training...')
        self.mean_squared_err = self.threshold + 1
        while (self.mean_squared_err > self.threshold) and (self.epoch < self.max_epochs):
            self.epoch += 1
            errors = np.array([])
            for example in trainset:
                # positive phase:
                net.observe(example)
                pos_prods   = np.array(correlation(net.v, net.h))
                pos_vis_act = np.array(net.v)
                pos_hid_act = np.array(net.h)

                # negative phase:
                net.generate()
                neg_prods   = np.array(correlation(net.v, net.h))
                neg_vis_act = np.array(net.v)
                neg_hid_act = activation(np.dot(net.v, net.W) + net.b)

                # updates:
                net.W += net.eta * (pos_prods - neg_prods)
                net.a += net.eta * (pos_vis_act - neg_vis_act)
                net.b += net.eta * (pos_hid_act - neg_hid_act)
                errors = np.append(errors, squared_error(example, net.v))

            # error update:
            self.mean_squared_err = errors.mean()
            print('error [' + str(self.epoch) + ']:', self.mean_squared_err)

        print('done after', self.epoch, 'epochs.')
        print('-----------')
