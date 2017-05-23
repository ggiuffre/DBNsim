import numpy as np
import pickle
import os

from DBNlogic.train import CDTrainer
from DBNlogic.util import Configuration, sigmoid, activation



def base(name = ''):
    """Given the name of a network, return the default path to it."""
    return os.path.join(os.path.dirname(__file__), 'nets', name)




class DBN(list):
    """Deep Belief Network (DBN)."""

    def __init__(self, layers = [], name = 'untitled'):
        """Constructor for a DBN."""
        super().__init__(layers)
        self.name = name

    def observe(self, data):
        """Set the DBN state according to a particular input."""
        layer_data = data
        for rbm in self:
            layer_data = rbm.observe(layer_data)

    def generate(self):
        """Generate a sample from the current weights."""
        layer_data = self[-1].generate()
        for rbm in self[-2::-1]:
            rbm.h = layer_data
            layer_data = rbm.generate()
        return layer_data

    def evaluate(self, data):
        """Set the DBN state according to a particular input
        and try to reconstruct that input."""
        self.observe(data)
        return self.generate()

    def learn(self, trainset, config = Configuration()):
        """Learn from a particular dataset."""
        for rbm in self:
            print('trainset shape:', trainset.shape)
            trainer = CDTrainer(rbm, config = config)
            for curr_error in trainer.run(trainset):
                yield {'rbm': self.index(rbm), 'err': curr_error}
            trainset = np.array(trainer.next_rbm_data)
    
    def save(self):
        """Save the network weights to a Pickle file."""
        net_file = base(self.name + '.pkl')
        pickle.dump(self[:], open(net_file, 'wb'))

    @staticmethod
    def load(name):
        """Load a network from a Pickle file."""
        net_file = base(self.name + '.pkl')
        return DBN(pickle.load(open(net_file, 'rb')), name = name)



class RBM:
    """Restricted Boltzmann Machine (RBM)."""

    def __init__(self, vis_size, hid_size):
        """Constructor for a RBM."""
        self.v = np.zeros((vis_size, 1)) # visible units
        self.a = np.zeros((vis_size, 1)) # visible biases
        self.h = np.zeros((hid_size, 1)) # hidden units
        self.b = np.zeros((hid_size, 1)) # hidden biases
        self.W = np.random.randn(hid_size, vis_size) * 0.1 # weights

    def observe(self, data):
        """Set the RBM state according to a particular input."""
        self.v = np.array(data).reshape(-1, 1) # vertical array
        hid_probs = sigmoid(np.dot(self.W, self.v) + self.b) # (W * v + b)
        self.h = activation(hid_probs)
        return hid_probs

    def generate(self):
        """Generate a sample from the current weights."""
        self.v = sigmoid(np.dot(self.W.T, self.h) + self.a) # (W' * h + a)
        return self.v

    def evaluate(self, data):
        """Set the RBM state according to a particular input
        and try to reconstruct that input."""
        self.observe(data)
        return self.generate()
