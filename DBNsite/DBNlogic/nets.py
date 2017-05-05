import numpy as np
import pickle

from train import CDTrainer
from util import sigmoid, activation



class DBN(list):
    """Deep Belief Network."""

    def __init__(self, layers, name = 'untitled'):
        super().__init__(layers)
        self.name = name

    def observe(self, data):
        """Set the DBN state according to a particular input."""
        layer_data = data
        for rbm in self:
            rbm.observe(layer_data)
            # TODO: <<<<<<<<<<<<<<<<
            layer_data = rbm.h # <<< ma è meglio usare le probs anziché le attivazioni

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

    def learn(self, trainset, threshold = 0.05, max_epochs = 10, batch_sz = 1):
        train_layer = trainset
        for rbm in self:
            trainer = CDTrainer(rbm, max_epochs = max_epochs, batch_sz = batch_sz)
            trainer.run(train_layer)
            train_layer = rbm.h.reshape(-1, 1) # TODO (stub)
    
    def save(self):
        net_file = 'nets/' + self.name + '.pkl'
        pickle.dump(self[:], open(net_file, 'wb'))

    @staticmethod
    def load(name):
        net_file = 'nets/' + name + '.pkl'
        return DBN(pickle.load(open(net_file, 'rb')), name = name)



class RBM:
    """Restricted Boltzmann Machine."""

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
        self.h = activation(sigmoid(np.dot(self.W, self.v) + self.b)) # (W * v + b)

    def generate(self):
        """Generate a sample from the current weights."""
        self.v = sigmoid(np.dot(self.W.T, self.h) + self.a) # (W' * h + a)
        return self.v

    def evaluate(self, data):
        """Set the RBM state according to a particular input
        and try to reconstruct that input."""
        self.observe(data)
        return self.generate()
