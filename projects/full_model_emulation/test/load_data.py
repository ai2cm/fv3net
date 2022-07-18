import torch
import numpy as np
import pandas as pd



def load_data(file_path, len_train, len_val):
    df = pd.read_csv(file_path, header=None).values.astype(float)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]
    return train, val, test


def data_transform(data, n_his, n_pred, lead,device):
    # produce data slices for training and testing
    if len(np.shape(data))==2:
        n_route = data.shape[1]
        l = len(data)
        num = l-n_his-n_pred
        x = np.zeros([num, 1, n_his, n_route])
        y = np.zeros([num, n_route])
        
        cnt = 0
        for i in range(l-n_his-n_pred):
            head = i
            tail = i + n_his
            x[cnt, :, :, :] = data[head: tail].reshape(1, n_his, n_route)
            y[cnt] = data[tail + n_pred - 1]
            cnt += 1
    elif len(np.shape(data))==3:
        n_route = data.shape[2]
        n_ch = data.shape[0]
        l = data.shape[1]
        num = l-n_his-n_pred
        x = np.zeros([num,n_ch, int(n_his/lead), n_route])
        y = np.zeros([num,n_ch,n_route])
        
        cnt = 0
        for i in range(l-n_his-n_pred):
            head = i
            tail = i + n_his
            x[cnt, :, :, :] = data[:,head: tail:lead,:].reshape(n_ch, int(n_his/lead), n_route)
            y[cnt] = data[:,tail + (lead*(n_pred-1))]
            cnt += 1


    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
