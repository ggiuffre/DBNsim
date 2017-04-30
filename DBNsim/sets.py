import csv
import pickle
import numpy as np
import os.path



class DataSet:
    """Dataset for training a neural network."""

    def __init__(self, csv_file):
        self.data = []
        with open(csv_file) as f:
            csv_data = csv.reader(f, delimiter = ',')
        for row in csv_data:
            example = {'label': None, 'image': []}
            example['label'] = int(row[0])
            example['image'] = [int(x) for x in row[1:]]
            self.data.append(example)



class MNIST(DataSet):
    """MNIST dataset."""

    def __init__(self):
        pkl_file = 'data/MNIST_labeled.pkl'
        csv_file = 'data/MNIST_labeled.csv'
        if (os.path.isfile(pkl_file)):
            self.data = pickle.load(open(pkl_file, 'rb'))
        else:
            super().__init__(csv_file)
            pickle.dump(self.data, open(pkl_file, 'wb'))
