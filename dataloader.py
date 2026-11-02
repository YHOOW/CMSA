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
    def __len__(self):import numpy as np
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



class Synthetic3d():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'synthetic3d.mat')
        self.Y = data['Y'].astype(np.int32)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 600
    def __getitem__(self, idx):
        x1 = np.array(self.V1[idx])
        x2 = np.array(self.V2[idx])
        x3 = np.array(self.V3[idx])
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Caltech20():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Caltech101-20.mat')
        self.Y = data['Y'].astype(np.int32)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
        self.V4 = data['X'][0][3].astype(np.float32)
        self.V5 = data['X'][0][4].astype(np.float32)
        self.V6 = data['X'][0][5].astype(np.float32)
    def __len__(self):
        return 2386
    def __getitem__(self, idx):
        x1 = np.array(self.V1[idx])
        x2 = np.array(self.V2[idx])
        x3 = np.array(self.V3[idx])
        x4 = np.array(self.V4[idx])
        x5 = np.array(self.V5[idx])
        x6 = np.array(self.V6[idx])
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
class CCV(Dataset):
    def __init__(self, path):
        scaler = MinMaxScaler()

        self.data1 = np.load(path+'STIP.npy').astype(np.float32)#'STIP.npy'
        # scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32) #'SIFT.npy'
        self.data2 = scaler.fit_transform(self.data2)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)#'MFCC.npy'原来的
        # self.data1 = scaler.fit_transform(self.data1)
        # self.data2 = scaler.fit_transform(self.data2)
        # self.data3 = scaler.fit_transform(self.data3)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
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
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]  #原[40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]  #[5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "Synthetic3d":
        dataset = Synthetic3d('./data/')
        dims = [ 3, 3, 3]
        view = 3
        data_size = 600
        class_num = 3
    elif dataset == "Caltech20":
        dataset = Caltech20('./data/')
        dims = [48, 40, 254, 1984, 512, 928]  #原数据[48, 40, 254, 1984, 512, 928]
        view = 6
        data_size = 2386
        class_num = 20
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
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
