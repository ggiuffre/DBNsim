import numpy as np

from plot import WeightsPlotter
from util import sigmoid, activation, squared_error



class CDTrainer:
    """Contrastive Divergence trainer for RBMs."""

    def __init__(self, net, threshold = 0.05, max_epochs = 100):
        """Constructor for a RBM."""
        self.net = net               # the learner
        self.mean_squared_err = None # current training error
        self.threshold = threshold   # target error
        self.epoch = 0               # current training epoch
        self.max_epochs = max_epochs # max n. of training epochs
        self.learn_rate = 0.6        # learning rate
        self.momentum = 0.8          # learning momentum
        self.batch_size = 1          # size of a (mini) batch

    def run(self, trainset):
        """Learn from a particular training set."""
        net      = self.net        # for readability
        batch_sz = self.batch_size # for readability
        print('-----------')
        print('training...')
        W_update = np.zeros(net.W.shape)
        a_update = np.zeros(net.a.shape)
        b_update = np.zeros(net.b.shape)
        self.mean_squared_err = self.threshold + 1 # for entering the while loop
        while (self.mean_squared_err > self.threshold) and (self.epoch < self.max_epochs):
            self.epoch += 1
            errors = np.array([])
            for batch_n in range(int(len(trainset) / batch_sz)):
                examples = np.array(trainset[batch_n : batch_n + batch_sz]).T
                data = examples

                # positive phase:
                hid_probs   = sigmoid(np.dot(net.W, data) + net.b.repeat(batch_sz, axis = 1))
                hid_states  = activation(hid_probs)
                pos_corr    = np.dot(hid_probs, data.T) # vis-hid correlation (+)
                pos_vis_act = data
                pos_hid_act = hid_probs

                # negative phase:
                vis_probs   = sigmoid(np.dot(net.W.T, hid_states) + net.a)
                data        = activation(vis_probs)
                hid_probs   = sigmoid(np.dot(net.W, data) + net.b)
                neg_corr    = np.dot(hid_probs, data.T) # vis-hid correlation (-)
                neg_vis_act = data
                neg_hid_act = hid_probs

                # updates:
                W_update = self.momentum * W_update + self.learn_rate * (pos_corr - neg_corr)
                a_update = self.momentum * a_update + self.learn_rate * (pos_vis_act - neg_vis_act)
                b_update = self.momentum * b_update + self.learn_rate * (pos_hid_act - neg_hid_act)
                net.W += W_update
                net.a += a_update
                net.b += b_update
                errors = np.append(errors, squared_error(examples, data))

            # error update:
            self.mean_squared_err = errors.mean()
            print('error [epoch ' + str(self.epoch) + ']:', self.mean_squared_err)
            ### yield self.mean_squared_err

        print('done after', self.epoch, 'epochs.')
        print('-----------')
