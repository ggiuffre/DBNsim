from __future__ import print_function

import os
import numpy as np
import pickle
import gzip as gz
import urllib2 as url

SCIPY_AVAILABLE = True
try:
    from scipy.io import savemat, loadmat
except ImportError:
    SCIPY_AVAILABLE = False



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
            pickle.dump(self.data, open(filename, 'wb'), protocol = 2)
        elif extension == '.mat':
            if SCIPY_AVAILABLE:
                savemat(filename, {'data': self.data})
            else:
                print("""you don't have the scipy package installed.
                    Install it with `pip install --user scipy` or, 
                    if you can't, download a precompiled binary""")

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
        return np.loadtxt(path, dtype = np.float32, delimiter = delimiter)

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
            data = pickle.load(f)
            if type(data) != np.ndarray:
                data = np.array(data)
        return data

    @staticmethod
    def fromMatlab(path):
        """Return a Numpy array of training examples from
        a Matlab file containing a variable called 'data'."""
        if SCIPY_AVAILABLE:
            print('loading data from Matlab file...')
            return loadmat(path, variable_names = ['data'])['data']
        else:
            print("""you don't have the scipy package installed.
                Install it with `pip install --user scipy` or, 
                if you can't, download a precompiled binary""")
            return np.array([])

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
        return frozenset([os.path.splitext(f)[0] for f in os.listdir(path) if (os.path.splitext(f)[1] in ['.pkl', '.csv', '.mat'])])



def get14x14MNIST():
    """Construct a 14x14 downsampling of the MNIST dataset."""
    mnist = DataSet.fromWhatever('MNIST')
    data = mnist.data.reshape(60000, 28, 28)
    data = data[:, ::2, ::2]
    data = data.reshape(60000, 14 * 14)
    return data

def get7x7MNIST():
    """Construct a 7x7 downsampling of the MNIST dataset."""
    mnist = DataSet.fromWhatever('MNIST')
    data = mnist.data.reshape(60000, 28, 28)
    data = data[:, ::4, ::4]
    data = data.reshape(60000, 7 * 7)
    return data

def convertMNIST(dataFile, labelsFile = None, outputFile = 'MNIST.csv'):
    """
    Convert the MNIST dataset from the original
    'ubyte' format to a CSV file.

    For example, convertMNIST('train-images-idx3-ubyte',
    'train-labels-idx1-ubyte', 'MNIST.csv') converts
    'train-images-idx3-ubyte' (with labels in
    'train-labels-idx1-ubyte') to 'MNIST.csv'.

    Or, if you don't want the labels in the CSV file, use
    convertMNIST('train-images-idx3-ubyte', None, 'MNIST.csv').
    """
    f = open(dataFile, 'rb')
    l = None
    if labelsFile != None:
        l = open(labelsFile, 'rb')
    o = open(outputFile, 'w')

    f.read(16)
    if labelsFile != None:
        l.read(8)
    images = []

    for i in range(60000):
        img = []
        if l != None:
            img = [ord(l.read(1))]
        for j in range(28 * 28):
            img.append(ord(f.read(1)))
        images.append(img)

    for img in images:
        o.write(','.join(str(pix) for pix in img) + '\n')

    f.close()
    if labelsFile != None:
        l.close()
    o.close()

def downloadMNIST(outputFile = full('MNIST.csv')):
    """Get the MNIST dataset from Yann LeCunn's
    web site and store it as a CSV file."""
    resource = url.urlopen('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz').read()
    archive = gz.open('MNIST_ubyte.gz', 'wb')
    archive.write(resource)
    with gz.open('MNIST_ubyte.gz', 'rb') as archive:
        file = open('MNIST_ubyte', 'wb')
        file.write(archive.read())
        file.close()
    #os.remove('MNIST_ubyte.gz')
    convertMNIST('MNIST_ubyte', None, outputFile)
    os.remove('MNIST_ubyte')
    data = DataSet.fromCSV(outputFile) / 255.0
    DataSet(data).save(outputFile[:-3] + 'pkl')
