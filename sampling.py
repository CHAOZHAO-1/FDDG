import h5py
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


import copy
import torch
import numpy as np
from scipy.fftpack import fft
import scipy.io as scio


def wgn(x, snr):
    Ps = np.sum(abs(x)**2,axis=1)/len(x)
    Pn = Ps/(10**((snr/10)))
    row,columns=x.shape
    Pn = np.repeat(Pn.reshape(-1,1),columns, axis=1)

    noise = np.random.randn(row,columns) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise


def zscore(Z):
    Zmax, Zmin = Z.max(axis=1), Z.min(axis=1)
    Z = (Z - Zmin.reshape(-1,1)) / (Zmax.reshape(-1,1) - Zmin.reshape(-1,1))
    return Z


def min_max(Z):
    Zmin = Z.min(axis=1)

    Z = np.log(Z - Zmin.reshape(-1, 1) + 1)
    return Z





def get_dataset(args,client):

    root_path = '/home/zhaochao/research/DTL/data/' + args.dataset + 'data' + str(args.class_num) + '.mat'

    data = scio.loadmat(root_path)

    train_loaders = {}


    for k in range(3):
        if args.fft1 == True:
            train_fea = zscore((min_max(abs(fft(data[client[k]]))[:, 0:512])))
        if args.fft1 == False:
            train_fea = zscore(data[client[k]])
    #
        train_label = torch.zeros((800 * args.class_num))
        for i in range(800 * args.class_num):
            train_label[i] = i // 800

        print(train_fea.shape)
        print(train_label.shape)
    # #
        train_label = train_label.long()
        train_fea = torch.from_numpy(train_fea)
        train_fea = torch.tensor(train_fea, dtype=torch.float32)
        data_s = torch.utils.data.TensorDataset(train_fea, train_label)
        train_loaders[k] = torch.utils.data.DataLoader(data_s, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                       num_workers=args.workers, pin_memory=args.pin)

    #
    if args.fft1 == True:
        test_fea = zscore((min_max(abs(fft(data[client[3]]))[:, 0:512])))
    if args.fft1 == False:
        test_fea = zscore(data[client[3]])

    test_label = torch.zeros((200 * args.class_num))
    for i in range(200 * args.class_num):
        test_label[i] = i // 200



    test_label = test_label.long()
    test_fea = torch.from_numpy(test_fea)
    test_fea = torch.tensor(test_fea, dtype=torch.float32)
    data_t = torch.utils.data.TensorDataset(test_fea, test_label)
    test_loader = torch.utils.data.DataLoader(data_t, batch_size=800, shuffle=True, drop_last=False,
                                               num_workers=args.workers, pin_memory=args.pin)

    return train_loaders, test_loader