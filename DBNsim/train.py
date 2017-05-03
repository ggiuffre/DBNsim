import numpy as np
import math
import random



def sigmoid(a):
    """Return the element-wise sigmoid of a vector."""
    ones = np.ones(a.shape) # array of ones
    return ones / (ones + np.exp(-a))

def activation(a):
    """Return the element-wise binary activation of a vector."""
    return sigmoid(a) > np.random.uniform(size = a.shape)

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
                hid_probs   = np.dot(example, net.W) + net.b
                net.h       = activation(hid_probs)
                pos_corr    = np.dot(example.T, hid_probs)
                pos_vis_act = np.array(example)
                pos_hid_act = np.array(hid_probs)

                # negative phase:
                vis_probs   = np.dot(net.W, net.h) + net.a
                net.v       = activation(vis_probs)
                hid_probs   = np.dot(net.v, net.W) + net.b
                neg_corr    = np.dot(net.v.T, hid_probs)
                neg_vis_act = np.array(net.v)
                neg_hid_act = activation(hid_probs)

                # updates:
                net.W += net.eta * (pos_corr - neg_corr)
                net.a += net.eta * (pos_vis_act - neg_vis_act)
                net.b += net.eta * (pos_hid_act - neg_hid_act)
                errors = np.append(errors, squared_error(example, net.v))

            # error update:
            self.mean_squared_err = errors.mean()
            print('error [epoch ' + str(self.epoch) + ']:', self.mean_squared_err)
            ### yield self.mean_squared_err

        print('done after', self.epoch, 'epochs.')
        print('-----------')
