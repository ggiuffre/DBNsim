import numpy as np
import pickle
import os



def exists(path):
    """Return whether a resource exists."""
    return os.path.isfile(path)

def full(name = ''):
    """Given the name of a dataset, return the default path to it."""
    return os.path.join(os.path.dirname(__file__), 'data', name)



class DataSet:
    """Dataset for training a neural network."""

    def __init__(self, data):
        self.data = data

    @staticmethod
    def fromCSV(path, delimiter = ','):
        print('loading data from CSV...')
        return np.genfromtxt(path, delimiter = delimiter)

    @staticmethod
    def fromPickle(path):
        print('loading data from pickle...')
        return np.array(pickle.load(open(path, 'rb')))

    @classmethod
    def fromWhatever(cls, name):
        full_path = full(name)
        if exists(full_path + '.pkl'):
            return cls.fromPickle(full_path + '.pkl')
        if exists(full_path + '.csv'):
            return cls.fromCSV(full_path + '.csv')
        return np.array([])

    @staticmethod
    def allSets(path = full()):
        return set([os.path.splitext(f)[0] for f in os.listdir(path) if (f.endswith('.pkl') or f.endswith('.csv'))])



class MNIST(DataSet):
    """MNIST dataset."""

    def __init__(self):
        pkl_file = full('MNIST_labeled.pkl')
        csv_file = full('MNIST_labeled.csv')
        if (exists(pkl_file)):
            self.data = DataSet.fromPickle(pkl_file)
        else:
            self.data = DataSet.fromCSV(csv_file)[:, 1:] / 255.0
            pickle.dump(self.data, open(pkl_file, 'wb'))



class SmallerMNIST(MNIST):
    """A 7x7 downsampling of the MNIST dataset."""

    def __init__(self):
        pkl_file = full('MNIST_small.pkl')
        if (exists(pkl_file)):
            self.data = DataSet.fromPickle(pkl_file)
        else:
            super().__init__()
            self.data = self.data.reshape(60000, 28, 28)
            print('downsampling data...')
            self.data = np.array([image[::4, ::4] for image in self.data])
            self.data = self.data.reshape(60000, 49)
            pickle.dump(self.data, open(pkl_file, 'wb'))
