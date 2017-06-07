import numpy as np
import pickle
from scipy.io import savemat, loadmat
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
        """Construct a dataset from an array of examples."""
        self.data = data

    def save(self, filename):
        """Save the dataset to a CSV, Pickle or Matlab file,
        based on the extension of the target filename."""
        name, extension = os.path.splitext(filename)
        if extension == '.csv':
            np.savetxt(filename, self.data, delimiter = ',')
        elif extension == '.pkl':
            pickle.dump(self.data, filename)
        elif extension == '.mat':
            savemat(filename, {'data': self.data})

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

    @staticmethod
    def fromMatlab(path):
        """Return a Numpy array of training examples from
        a Matlab file containing a variable called 'data'."""
        print('loading data from Matlab file...')
        return loadmat(path)['data']

    @classmethod
    def fromWhatever(cls, name):
        """Return a Numpy array of training examples from a
        Pickle, CSV, or Matlab file with the given name, depending
        on what is available (`name` doesn't have the extension)."""
        full_path = full(name)
        if exists(full_path + '.pkl'):
            return cls.fromPickle(full_path + '.pkl')
        elif exists(full_path + '.csv'):
            return cls.fromCSV(full_path + '.csv')
        elif exists(full_path + '.mat'):
            return cls.fromMatlab(full_path + '.mat')
        return np.array([])

    @staticmethod
    def allSets(path = full()):
        """Return a set with the filenames of all the
        available training datasets (without the extension)."""
        return set([os.path.splitext(f)[0] for f in os.listdir(path) if (os.path.splitext(f)[1] in ['.pkl', '.csv', '.mat'])])



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
