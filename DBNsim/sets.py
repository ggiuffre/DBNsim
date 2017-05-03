import csv
import pickle
import numpy as np
import os.path



class DataSet:
    """Dataset for training a neural network."""

    def __init__(self, data):
        self.data = data
        self.index = len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]

    @staticmethod
    def fromCSV(path, shape = None):
        print('loading data from CSV...')
        with open(path) as f:
            csv_data = csv.reader(f, delimiter = ',')
            data = np.array([[float(x) for x in row[1:]] for row in csv_data])
        if shape != None:
            data.reshape(shape)
        return data

    @staticmethod
    def fromPickle(path, shape = None):
        print('loading data from pickle...')
        data = np.array(pickle.load(open(path, 'rb')))
        if shape != None:
            data.reshape(shape)
        return data



class MNIST(DataSet):
    """MNIST dataset."""

    def __init__(self):
        pkl_file = 'data/MNIST_labeled.pkl'
        csv_file = 'data/MNIST_labeled.csv'
        if (os.path.isfile(pkl_file)):
            data = pickle.load(open(pkl_file, 'rb'))
        else:
            data = DataSet.fromCSV(csv_file, shape = (60000, 784))
            data = [[x / 255.0 for x in row] for row in data]
            pickle.dump(data, open(pkl_file, 'wb'))
        super().__init__(data)



class SmallerMNIST(DataSet):
    """A 7x7 downsampling of the MNIST dataset."""

    def __init__(self):
        pkl_file = 'data/MNIST_small_labeled.pkl'
        if (os.path.isfile(pkl_file)):
            data = pickle.load(open(pkl_file, 'rb'))
        else:
            data = np.array(MNIST())
            data = data.reshape(60000, 7, 7, 4, 4)
            data = np.array([[[group.mean() for group in piece] for piece in example] for example in data])
            data = data.reshape(60000, 49)
            pickle.dump(data, open(pkl_file, 'wb'))
        super().__init__(data)
