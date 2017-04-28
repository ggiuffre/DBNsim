import csv
import pickle
import numpy as np
import os.path



class TrainSet:
    """Training set."""

    def __init__(self, path):
        self.data = []
        csv_file = open(path)
        csv_data = csv.reader(csv_file, delimiter = ',')
        for row in csv_data:
            example = {'label': None, 'image': []}
            example['label'] = int(row[0])
            example['image'] = [int(x) for x in row[1:]]
            self.data.append(example)



class MNIST(TrainSet):
    """MNIST training set."""

    def __init__(self):
        pkl_data = 'data/MNIST_labeled.pkl'
        csv_data = 'data/MNIST_labeled.csv'
        if (os.path.isfile(pkl_data)):
            self.data = pickle.load(open(pkl_data, 'rb'))
        else:
            super().__init__(csv_data)
            pickle.dump(self.data, open(pkl_data, 'wb'))
