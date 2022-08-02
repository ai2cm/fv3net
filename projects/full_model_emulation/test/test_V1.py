import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import SAGEConv
from dgl.nn.pytorch import GraphConv, ChebConv, GATConv
import numpy as np
import dask.diagnostics
import fsspec
import xarray as xr
import matplotlib.pyplot as plt
import networkx as nx
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import warnings
import time
import select as sl
import pickle
from load_data import *
from utils import *
from Newmodel import *
import wandb
from fv3net.artifacts.resolve_url import resolve_url
from vcm import get_fs

lead=1

lead2=1

day=1
control_str='TATNTATN'#'TNSTTNST' #'TNTSTNTST'
epochs=10
num_heads=2 

variableList=['h500','h200','h850']
TotalSamples=8500
Chuncksize=1000



lr=0.001
disablecuda ='store_true'
batch_size=10
window=24*day
pred_len=1
channels=[7, 16, 32, 64, 32, 128]
drop_prob = 0
out_feat=2

savemodelpath = (
    "weight_layer_"
    + control_str
    + "_lag"
    + str(lead)
    +"_lead"
    +str(lead2)
    + "_window"
    + str(window)
    + "_epochs_"
    + str(epochs)
    + ".pt"
)

BUCKET = "vcm-ml-experiments"
PROJECT = "full-model-emulation"

model_out_url = resolve_url(BUCKET, PROJECT, savemodelpath)
data_url = "gs://vcm-ml-scratch/ebrahimn/2022-07-02/experiment-1-y/fv3gfs_run/"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_url = "gs://vcm-ml-scratch/ebrahimn/2022-07-02/experiment-1-y/fv3gfs_run/" 
state_training_data = xr.open_zarr(
    fsspec.get_mapper(os.path.join(data_url, "atmos_dt_atmos.zarr"))
)
# state_training_data2 = xr.open_zarr(fsspec.get_mapper(os.path.join(data_url, 'sfc_dt_atmos.zarr')))
lat_lon_data = xr.open_zarr(
    fsspec.get_mapper(os.path.join(data_url, "state_after_timestep.zarr"))
)

landSea = xr.open_zarr(
    fsspec.get_mapper(
        "gs://vcm-ml-experiments/default/2022-05-09/baseline-35day-spec-sst/fv3gfs_run/state_after_timestep.zarr"
    )
)
landSea_Mask = landSea.land_sea_mask[1].load()
landSea_Mask = landSea_Mask[:, ::, ::].values.flatten()


lat=(lat_lon_data.latitude[1].load())
lon=(lat_lon_data.longitude[1].load())
lat=lat[:,::,::].values.flatten()
lon=lon[:,::,::].values.flatten()
# cosLat=np.expand_dims(np.cos(lat),axis=1)
# cosLon=np.expand_dims(np.cos(lon),axis=1)
# sinLat=np.expand_dims(np.sin(lat),axis=1)
# sinLon=np.expand_dims(np.sin(lon),axis=1)
cosLat=np.cos(lat)
cosLon=np.cos(lon)
sinLat=np.sin(lat)
sinLon=np.sin(lon)
for i in range(3):
        if i==0:
            sinLon=torch.tensor(sinLon).unsqueeze(0).repeat(int(window/lead),1)
            cosLon=torch.tensor(cosLon).unsqueeze(0).repeat(int(window/lead),1)
            sinLat=torch.tensor(sinLat).unsqueeze(0).repeat(int(window/lead),1)
            cosLat=torch.tensor(cosLat).unsqueeze(0).repeat(int(window/lead),1)
            landSea_Mask=torch.tensor(landSea_Mask).unsqueeze(0).repeat(int(window/lead),1)
        elif i==1:
            sinLon=(sinLon).unsqueeze(0).repeat(1,1,1)
            cosLon=(cosLon).unsqueeze(0).repeat(1,1,1)
            sinLat=(sinLat).unsqueeze(0).repeat(1,1,1)
            cosLat=(cosLat).unsqueeze(0).repeat(1,1,1)
            landSea_Mask=(landSea_Mask).unsqueeze(0).repeat(1,1,1)

        elif i==2:
            sinLon=(sinLon).unsqueeze(0).repeat(batch_size,1,1,1)
            cosLon=(cosLon).unsqueeze(0).repeat(batch_size,1,1,1)
            sinLat=(sinLat).unsqueeze(0).repeat(batch_size,1,1,1)
            cosLat=(cosLat).unsqueeze(0).repeat(batch_size,1,1,1)
            landSea_Mask=(landSea_Mask).unsqueeze(0).repeat(batch_size,1,1,1)

exteraVar=torch.cat((sinLon, sinLat,cosLon,cosLat,landSea_Mask), 1).to(device)

num_nodes=len(lon)
print(f"numebr of grids: {num_nodes}")


all_indices=np.random.permutation(np.arange(start=0, stop=int(TotalSamples/Chuncksize)))



for ss in np.arange(0,int(TotalSamples/Chuncksize)):

    Z500train=state_training_data[variableList[0]].isel(time=slice((ss*Chuncksize),(ss+1)*Chuncksize))
    T2mtrain1=state_training_data[variableList[1]].isel(time=slice((ss*Chuncksize),(ss+1)*Chuncksize))
    T2mtrain2=state_training_data[variableList[2]].isel(time=slice((ss*Chuncksize),(ss+1)*Chuncksize))

    Z500train=np.swapaxes(Z500train.values, 1, 0)
    T2mtrain1=np.swapaxes(T2mtrain1.values, 1, 0)
    T2mtrain2=np.swapaxes(T2mtrain2.values, 1, 0)

    T2mtrain=T2mtrain1-T2mtrain2

    T2mtrain=T2mtrain.reshape(np.size(T2mtrain, 0), np.size(T2mtrain, 1)*np.size(T2mtrain, 2)*np.size(T2mtrain, 3))
    Z500train=Z500train.reshape(np.size(Z500train, 0), np.size(Z500train, 1)*np.size(Z500train, 2)*np.size(Z500train, 3))

    Zmean = np.mean(Z500train)
    Zstd = np.std(Z500train)

    Tmean = np.mean(T2mtrain)
    Tstd = np.std(T2mtrain)


    T2mtrain=(T2mtrain-Tmean)/Tstd
    Z500train=(Z500train-Zmean)/Zstd
    # Z500train=np.swapaxes(Z500train, 1, 0)
    # T2mtrain=np.swapaxes(T2mtrain, 1, 0)

    g = pickle.load(open("UpdatedGraph", 'rb'))

    T2mtrain=np.expand_dims(T2mtrain,axis=0)
    Z500train=np.expand_dims(Z500train,axis=0)

    dataSets=np.concatenate((Z500train,T2mtrain),axis=0)

    num_samples=np.size(dataSets,1)
    print(f"Total samples: {num_samples}")


    len_val = round(num_samples * 0.25)
    len_train = round(num_samples * 0.75)
    train = dataSets[:,: len_train]
    val = dataSets[:,len_train+14: len_train + len_val]

    x_train, y_train = data_transform(train, window, pred_len, lead, lead2, device)
    x_val, y_val = data_transform(val, window, pred_len, lead, lead2, device)
    # x_test, y_test = data_transform(test, n_his, n_pred, device)
    print('size of training dataset:',np.shape(x_train),' and ',np.shape(x_train))
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    val_iter = torch.utils.data.DataLoader(val_data, batch_size)
    # test_data = torch.utils.data.TensorDataset(x_test, y_test)
    # test_iter = torch.utils.data.DataLoader(test_data, batch_size)


    loss = nn.MSELoss()
    g = g.to(device)
    model = STGCN_WAVE(channels, int(window/lead), out_feat, num_nodes, g, drop_prob, device, num_heads, control_str).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    if ss==0:
        min_val_loss = np.inf


    for epoch in range(1, epochs + 1):
        l_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            exteraVar1=exteraVar[:x.size(0)]
            x=torch.cat((x, exteraVar1), 1).float()
            y_pred = model(x).view(len(x), out_feat, -1)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            print("section ",ss," epoch", epoch, ", train loss:", l.item())

        scheduler.step()
        val_loss = evaluate_model(model, loss, val_iter,exteraVar,out_feat)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), savemodelpath)
        print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)

        fs = get_fs(model_out_url)
        fs.put(savemodelpath, model_out_url)
        print(savemodelpath, model_out_url)



# best_model = STGCN_WAVE(channels, window, num_nodes, g, drop_prob, num_layers, device, control_str).to(device)
# best_model.load_state_dict(torch.load(savemodelpath))

# l = evaluate_model(best_model, loss, test_iter)
# MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
# print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
