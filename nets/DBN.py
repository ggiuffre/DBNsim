from __future__ import print_function
import numpy as np
import math
import random

from train.CDTrainer import CDTrainer



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

    def learn(self, trainset, threshold = 0.05):
        for rbm in self.layers:
            trainer = CDTrainer(rbm)
            trainer.run(trainset) # NO! solo uno stub...
