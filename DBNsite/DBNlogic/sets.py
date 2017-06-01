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

    def __init__(self, data, imgShape = None):
        """Construct a dataset from an array of examples."""
        self.data = data
        if imgShape == None:
            imgShape = (len(self.data[0]),)
        self.imgShape = imgShape

    @staticmethod
    def fromCSV(path, delimiter = ','):
        """
        Return a Numpy array of training examples
        from a CSV file.

        The file must be a sequence of examples, each
        one separated by a newline and containing
        `delimiter`-separated values.
        """
        print('loading data from CSV...')
        return np.genfromtxt(path, delimiter = delimiter)

    @staticmethod
    def fromPickle(path):
        """
        Return a Numpy array of training
        examples from a Pickle file.

        A Pickle file is heavier than an equivalent
        CSV file, but is loaded much faster by Python.
        """
        data = None
        print('loading data from pickle...')
        with open(path, 'rb') as f:
            data = np.array(pickle.load(f))
        return data

    @classmethod
    def fromWhatever(cls, name):
        """Return a Numpy array of training examples
        from a Pickle or CSV file with the given name
        (`name` doesn't have the extension), depending
        on what is available."""
        full_path = full(name)
        if exists(full_path + '.pkl'):
            return cls.fromPickle(full_path + '.pkl')
        if exists(full_path + '.csv'):
            return cls.fromCSV(full_path + '.csv')
        return np.array([])

    @staticmethod
    def allSets(path = full()):
        """Return a set with the filenames of all the
        available training datasets (without the extension)."""
        return set([os.path.splitext(f)[0] for f in os.listdir(path) if (f.endswith('.pkl') or f.endswith('.csv'))])



class MNIST(DataSet):
    """The MNIST dataset."""

    def __init__(self):
        """Construct the MNIST dataset."""
        pkl_file = full('MNIST.pkl')
        csv_file = full('MNIST.csv')
        if (exists(pkl_file)):
            self.data = DataSet.fromPickle(pkl_file)
        else:
            self.data = DataSet.fromCSV(csv_file)[:, 1:] / 255.0
            with open(pkl_file, 'wb') as f:
                pickle.dump(self.data, f)
        self.imgShape = (28, 28)



class SmallerMNIST(MNIST):
    """A 7x7 downsampling of the MNIST dataset."""

    def __init__(self):
        """Construct a 7x7 downsampling of the MNIST dataset."""
        pkl_file = full('small_MNIST.pkl')
        if (exists(pkl_file)):
            self.data = DataSet.fromPickle(pkl_file)
        else:
            super().__init__()
            self.data = self.data.reshape(60000, 28, 28)
            print('downsampling data...')
            self.data = np.array([image[::4, ::4] for image in self.data])
            self.data = self.data.reshape(60000, 49)
            with open(pkl_file, 'wb') as f:
                pickle.dump(self.data, open(pkl_file, 'wb'))
        self.imgShape = (7, 7)
