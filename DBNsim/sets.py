import numpy as np
import pickle
import os.path



class DataSet:
    """Dataset for training a neural network."""

    def __init__(self, data):
        self.data = data

    @staticmethod
    def fromCSV(path, shape = None, delimiter = ','):
        print('loading data from CSV...')
        data = np.genfromtxt(path, delimiter = delimiter)
        if shape != None:
            data.reshape(shape)
        return data # return DataSet(data) ...?

    @staticmethod
    def fromPickle(path, shape = None):
        print('loading data from pickle...')
        data = np.array(pickle.load(open(path, 'rb')))
        if shape != None:
            data.reshape(shape)
        return data # return DataSet(data) ...?



class MNIST(DataSet):
    """MNIST dataset."""

    def __init__(self):
        pkl_file = 'data/MNIST_labeled.pkl'
        csv_file = 'data/MNIST_labeled.csv'
        if (os.path.isfile(pkl_file)):
            self.data = DataSet.fromPickle(pkl_file)
        else:
            self.data = DataSet.fromCSV(csv_file)[:, 1:] / 255.0
            pickle.dump(self.data, open(pkl_file, 'wb'))



class SmallerMNIST(MNIST):
    """A 7x7 downsampling of the MNIST dataset."""

    def __init__(self):
        pkl_file = 'data/MNIST_small_labeled.pkl'
        if (os.path.isfile(pkl_file)):
            self.data = DataSet.fromPickle(pkl_file)
        else:
            super().__init__()
            self.data = self.data.reshape(60000, 28, 28)
            print('downsampling data...')
            self.data = np.array([image[::4, ::4] for image in self.data])
            self.data = self.data.reshape(60000, 49)
            pickle.dump(self.data, open(pkl_file, 'wb'))
