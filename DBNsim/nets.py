import numpy as np
import math
import random

from train import CDTrainer



def sigmoid(a):
    """Return the sigmoid of a vector."""
    return [1.0 / (1.0 + math.exp(-x)) for x in a]

def activation(a):
    """Return a binary activation vector."""
    return [x > random.uniform(0, 1) for x in sigmoid(a)]

def correlation(a, b):
    """Return the binary correlation between 2 vectors."""
    return [[i * j for j in b] for i in a]



class DBN:
    """Deep Belief Network."""

    def __init__(self, layers):
        self.layers = layers

    def set(self, data):
        """Set the DBN state according to a particular input."""
        layer_data = data
        for rbm in self.layers:
            rbm.set(layer_data)
            layer_data = rbm.h # <<< ma è meglio usare le probs anziché le attivazioni

    def get(self):
        """Generate a sample from the current weights."""
        layer_data = self.layers[-1].get()
        for rbm in self.layers[-2::-1]:
            rbm.h = layer_data
            layer_data = rbm.get()
        return layer_data

    def evaluate(self, data):
        """Set the DBN state according to a particular input
        and reconstruct the input."""
        self.set(data)
        return self.get()

    def learn(self, trainset, threshold = 0.05, max_epochs = 10):
        for rbm in self.layers:
            trainer = CDTrainer(rbm)
            trainer.run(trainset) # NO! solo uno stub...



class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, vis_size, hid_size):
        """Constructor for a RBM."""
        self.v   = np.zeros(vis_size) # visible units
        self.a   = np.zeros(vis_size) # visible biases
        self.h   = np.zeros(hid_size) # hidden units
        self.b   = np.zeros(hid_size) # hidden biases
        self.W   = np.random.randn(vis_size, hid_size) * 0.01 # weights
        self.eta = 0.5 # learning rate
        self.max_epochs = 100 # max n. of epochs for training

    def __str__(self):
        """Return the weights representing the RBM."""
        return str(self.W) + '\n' + str(self.a) + '\n' + str(self.b)

    def __repr__(self):
        """Return the dimensions of the RBM."""
        return 'RBM(' + str(len(self.v)) + ', ' + str(len(self.h)) + ')'

    def set(self, data):
        """Set the RBM state according to a particular input."""
        self.v = data
        self.h = activation(np.dot(self.v, self.W) + self.b) # (vW + b)

    def get(self):
        """Generate a sample from the current weights."""
        self.v = activation(np.dot(self.W, self.h) + self.a) # (Wh + a)
        return self.v

    def evaluate(self, data):
        """Set the RBM state according to a particular input
        and reconstruct the input."""
        self.set(data)
        return self.get()

    def learn(self, trainset, threshold = 0.06):
        """Learn from a particular training set."""
        print('-----------')
        print('training...')
        done  = False
        epoch = 0
        while (not done) and (epoch < self.max_epochs):
            epoch += 1
            for example in trainset:
                # positive phase:
                self.set(example)
                pos_prods   = np.array(correlation(self.v, self.h))
                pos_vis_act = np.array(self.v)
                pos_hid_act = np.array(self.h)

                # negative phase:
                self.get()
                neg_prods   = np.array(correlation(self.v, self.h))
                neg_vis_act = np.array(self.v)
                neg_hid_act = activation(np.dot(self.v, self.W) + self.b)

                # update:
                self.W += self.eta * (pos_prods - neg_prods)
                self.a += self.eta * (pos_vis_act - neg_vis_act)
                self.b += self.eta * (pos_hid_act - neg_hid_act)

            # error calculation:
            done = all([all([abs(x) < threshold for x in row]) for row in pos_prods - neg_prods])

        print('done after', epoch, 'epochs.')
        print('-----------')
