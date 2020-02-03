import numpy as np
import os
import pandas

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import csv

class Dataset():
    def __init__(self, name, N, D, type, data_path='/data/'):
        self.data_path = data_path
        self.name, self.N, self.D = name, N, D
        assert type in ['regression', 'classification', 'multiclass']
        self.type=type

    def csv_file_path(self, name):
        return '{}{}.csv'.format(self.data_path, name)

    def read_data(self):
        data = pandas.read_csv(self.csv_file_path(self.name), header=None, 
                delimiter=',').values
        return {'X':data[:, :-1], 'Y':data[:, -1, None]}

    def download_data(self):
        NotImplementedError

    def get_data(self, seed=0, split=0, prop=0.9):
        path = self.csv_file_path(self.name)
        if not os.path.isfile(path):
            self.download_data()

        full_data = self.read_data()
        split_data = self.split(full_data, seed, split, prop)
        split_data = self.normalise(split_data, 'X')

        if self.type is 'regression':
            split_data = self.normalise(split_data, 'Y')

        return split_data

    def split(self, full_data, seed, split, prop):
        ind = np.arange(self.N)

        np.random.seed(seed + split)
        np.random.shuffle(ind)

        n = int(self.N * prop)

        X = full_data['X'][ind[:n], :]
        Xs = full_data['X'][ind[n:], :]

        Y = full_data['Y'][ind[:n], :]
        Ys = full_data['Y'][ind[n:], :]

        return {'X':X, 'Xs':Xs, 'Y':Y, 'Ys':Ys}
