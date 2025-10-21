import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from sklearn.preprocessing import MinMaxScaler


class Cifar100():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = np.array(self.V1[idx])
        x2 = np.array(self.V2[idx])
        x3 = np.array(self.V3[idx])
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Prokaryotic():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Prokaryotic.mat')
        self.Y = data['Y'].astype(np.int32)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 551
    def __getitem__(self, idx):
        x1 = np.array(self.V1[idx])
        x2 = np.array(self.V2[idx])
        x3 = np.array(self.V3[idx])
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()



def load_data(dataset):
    if dataset == "Cifar100":
        dataset = Cifar100('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100
    elif dataset == "Prokaryotic":
        dataset = Prokaryotic('./data/')
        dims = [ 438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
