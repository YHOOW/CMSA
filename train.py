import torch
from network import CMSA
import torch.nn as nn
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import os
from torch.utils.data import Dataset, DataLoader, Subset

# Prokaryotic
# Cifar100

Dataname = 'Prokaryotic'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--rec_epochs", default=10)
parser.add_argument("--fine_tune_epochs", default=200)
parser.add_argument("--low_feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "Cifar100":
    args.rec_epochs = 5
    args.fine_tune_epochs = 30
    seed = 10
if args.dataset == "Prokaryotic":
    args.rec_epochs = 20
    args.fine_tune_epochs = 20
    seed = 7

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)
dataset, dims, view, data_size, class_num = load_data(args.dataset)

class SingleViewDataset(Dataset):
    def __init__(self, dataset, view):
        """
        dataset: 原始多视角数据集
        view: 当前视角的索引
        """
        self.dataset = dataset
        self.view = view

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        xs, labels, additional_info = self.dataset[idx]
        # 仅返回当前视角的数据
        return xs[self.view], labels, additional_info
def pre_train(epoch, view_loader):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(view_loader):
        xs = xs.to(device)
        optimizer.zero_grad()
        xrs, _= model(xs)
        loss = mse(xs, xrs)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(view_loader)))

def triu(X):
    return torch.sum(torch.triu(X, diagonal=1))

def fine_tune(epoch, view_loader , previous_total_CF, previous_total_commonz):
    T = 1.5
    loss_total = 0.
    mes = torch.nn.MSELoss()
    total_CF = []
    total_commonz = []
    for batch_idx, (xs, ys, _) in enumerate(view_loader):
        xs = xs.to(device)
        optimizer.zero_grad()
        xrs, _= model(xs)
        commonz, S = model.SDF(xs)
        if v > 0:
            start_idx = batch_idx * 256
            end_idx = (batch_idx + 1) * 256
            previous_CF_part = previous_total_CF[start_idx:end_idx, :].to(device)
            SCF = (previous_CF_part @ previous_CF_part.t()).to(device)
            print(f"previous_S_part shape: {SCF.shape}")
            S = S.to(device)

            weight = nn.Parameter(torch.full((2,), 1.0, device=device), requires_grad=True)
            weight = F.softmax(weight)
            weight = torch.pow(weight, -1)

            W = torch.zeros((256, 256, 2))
            W[:, :, 0] = SCF
            W[:, :, 1] = S
            W = W.to(device)
            S = torch.matmul(W, weight)
        if v > 0:
            CF = model.apply_multihead_attention(S)
        else:
            CF = model.apply_multihead_attention(S)
        loss_cc = criterion.Structure_guided_Contrastive_Loss(CF, commonz, S)
        loss_gr1 = (mes(xs, xrs))
        if v > 0:
            start_idx = batch_idx * 256
            end_idx = (batch_idx + 1) * 256
            previous_total_commonz_part = previous_total_commonz[start_idx:end_idx, :].to(device)
            distillation_loss = torch.nn.functional.kl_div(F.log_softmax(commonz / T, dim=1),
                                                            F.softmax(previous_total_commonz_part / T, dim=1),
                                                            reduction='batchmean') * (T * T)
            loss_total = loss_total + loss_gr1 + loss_cc + 0.01*distillation_loss
        else:
            loss_total = loss_total+ loss_gr1 + loss_cc
        total_CF.append(CF)
        total_commonz.append(commonz)
    total_CF = torch.cat(total_CF, dim=0)
    total_commonz = torch.cat(total_commonz, dim=0)
    loss_or = 2 / (class_num * (class_num - 1)) * triu(torch.t(total_CF) @ total_CF)
    loss_total = loss_total + loss_or
    loss_total.backward()
    optimizer.step()
    optimizer.zero_grad()
    avg_loss = loss_total / len(view_loader)
    print(f'Epoch [{epoch}/{epoch}], Loss: {avg_loss}')
    return total_CF, total_commonz

if not os.path.exists('./models'):
    os.makedirs('./models')

for v in range(view):
    print(f'Training for View {v + 1}')
    view_dataset = SingleViewDataset(dataset, v)
    input_size = dims[v]
    model = CMSA(class_num ,1, input_size, args.low_feature_dim, args.high_feature_dim, device)  # 只需要处理单视图
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, args.temperature_f, device).to(device)
    if v == 0:
        previous_total_CF = None
        previous_total_commonz= None
    epoch = 1
    while epoch <= args.rec_epochs:
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)

        view_loader = DataLoader(
            view_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=True,
        )

        pre_train(epoch, view_loader)
        epoch += 1

    # 微调阶段
    while epoch <= args.rec_epochs + args.fine_tune_epochs:

        generator = torch.Generator()
        generator.manual_seed(seed + epoch)

        view_loader = DataLoader(
            view_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=True,
        )

        total_CF, total_commonz = fine_tune(epoch, view_loader , previous_total_CF, previous_total_commonz)  # 传入当前视图v

        if epoch == args.rec_epochs + args.fine_tune_epochs:
            valid(model, device, dataset, view, data_size, class_num, v, Dataname, epoch)
            state = model.state_dict()

            previous_total_CF = total_CF.detach()


            previous_total_commonz = total_commonz.detach()

            torch.save(state, './models/' + args.dataset + '_view_' + str(v) + '.pth')
            print(f'Saving model for View {v + 1}...')
        epoch += 1
