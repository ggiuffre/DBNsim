import csv
import pickle
import numpy as np
import os.path



class DataSet:
    """Dataset for training a neural network."""

    def __init__(self, csv_file, shape = None):
        with open(csv_file) as f:
            csv_data = csv.reader(f, delimiter = ',')
            self.data = np.array([[int(x) for x in row[1:]] for row in csv_data])
        if shape != None:
            self.data.reshape(shape)



class MNIST(DataSet):
    """MNIST dataset."""

    def __init__(self):
        pkl_file = 'data/MNIST_labeled.pkl'
        csv_file = 'data/MNIST_labeled.csv'
        if (os.path.isfile(pkl_file)):
            self.data = pickle.load(open(pkl_file, 'rb'))
        else:
            super().__init__(csv_file, shape = (60000, 784))
            pickle.dump(self.data, open(pkl_file, 'wb'))



class SmallerMNIST(MNIST):
    """A 7x7 downsampling of the MNIST dataset."""

    def __init__(self):
        pkl_file = 'data/MNIST_small_labeled.pkl'
        if (os.path.isfile(pkl_file)):
            self.data = pickle.load(open(pkl_file, 'rb'))
        else:
            super().__init__()
            self.data = self.data.reshape(60000, 7, 7, 4, 4)
            self.data = np.array([[[group.mean() for group in piece] for piece in example] for example in self.data])
            self.data = self.data.reshape(60000, 49)
            pickle.dump(self.data, open(pkl_file, 'wb'))
