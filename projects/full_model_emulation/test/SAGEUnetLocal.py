import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import SAGEConv
from dgl.nn.pytorch import NNConv
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
from utilsMPGNNUnet import *


lead=6

coarsenInd=3

control_str='SAGEUnet'#'TNSTTNST' #'TNTSTNTST'

print(control_str)

epochs=20

variableList=['h500','h200','h850']
TotalSamples=8500
Chuncksize=200
num_step=4


lr=0.001
disablecuda ='store_true'
batch_size=1
drop_prob = 0
out_feat=2

savemodelpath = (
    "weight_layer_"
    + control_str
    + "_lead"
    + str(lead)
    + "_epochs_"
    + str(epochs)
    +"MP_Block_"
    +str(num_step)
    + ".pt"
)


data_url = "gs://vcm-ml-scratch/ebrahimn/2022-07-02/experiment-1-y/fv3gfs_run/"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_url = "gs://vcm-ml-scratch/ebrahimn/2022-07-02/experiment-1-y/fv3gfs_run/" 
state_training_data = xr.open_zarr(
    fsspec.get_mapper(os.path.join(data_url, "atmos_dt_atmos.zarr")), consolidated=True
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
landSea_Mask = landSea_Mask[:, ::coarsenInd, ::coarsenInd].values.flatten()


lat=(lat_lon_data.latitude[1].load())
lon=(lat_lon_data.longitude[1].load())
lat=lat[:,::coarsenInd,::coarsenInd].values.flatten()
lon=lon[:,::coarsenInd,::coarsenInd].values.flatten()
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
            sinLon=torch.tensor(sinLon).unsqueeze(0).repeat(1,1)
            cosLon=torch.tensor(cosLon).unsqueeze(0).repeat(1,1)
            sinLat=torch.tensor(sinLat).unsqueeze(0).repeat(1,1)
            cosLat=torch.tensor(cosLat).unsqueeze(0).repeat(1,1)
            landSea_Mask=torch.tensor(landSea_Mask).unsqueeze(0).repeat(1,1)
        elif i==2:
            sinLon=(sinLon).unsqueeze(0).repeat(batch_size,1,1)
            cosLon=(cosLon).unsqueeze(0).repeat(batch_size,1,1)
            sinLat=(sinLat).unsqueeze(0).repeat(batch_size,1,1)
            cosLat=(cosLat).unsqueeze(0).repeat(batch_size,1,1)
            landSea_Mask=(landSea_Mask).unsqueeze(0).repeat(batch_size,1,1)

exteraVar=torch.cat((sinLon, sinLat,cosLon,cosLat,landSea_Mask), 1).to(device)
exteraVar=np.swapaxes(exteraVar,2, 1)
print(device)

num_nodes=len(lon)
print(f"numebr of grids: {num_nodes}")



g = pickle.load(open("UpdatedGraph_Neighbour10_Coarsen3", 'rb'))

edg=np.asarray(g.edges())
latInd=lat[edg[1]]
lonInd=lon[edg[1]]
latlon=[latInd.T,lonInd.T]
# latlon=np.swapaxes(latlon, 1, 0)
latlon=torch.from_numpy(np.swapaxes(latlon, 1, 0)).float()
latlon=latlon.to(device)


Zmean=5765.8457   #Z500mean=5765.8457, 
Zstd=90.79599   #Z500std=90.79599

Tmean=10643.382          #Thickmean=10643.382
Tstd=162.12427              #Thickstd=162.12427
valInde=0

print('loading model')

class UnetGraphSAGE(nn.Module):
    def __init__(self, g, in_feats, h_feats,out_feat,num_step):
        super(UnetGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, int(h_feats/2), 'mean')
        self.conv3 = SAGEConv(int(h_feats/2), int(h_feats/4), 'mean')
        self.conv4 = SAGEConv(int(h_feats/4), int(h_feats/4), 'mean')
        self.conv5 = SAGEConv(int(h_feats/2), int(h_feats/2), 'mean')
        self.conv6 = SAGEConv(h_feats, out_feat, 'mean')
        self.g=g
        self.num_step=num_step
    def forward(self, in_feat,exteraVar1):

        for _ in range(self.num_step):
            h = self.conv1(g, in_feat)
            h = F.relu(h)
            h = self.conv2(g, h)
            h = F.relu(h)
            h = self.conv3(g, h)
            h = F.relu(h)
            tuple = (self.conv4(self.g, h),h)
            h = torch.cat(tuple,dim=1)
            h = F.relu(h)
            tuple = (self.conv5(self.g, h),h)
            h = torch.cat(tuple,dim=1)
            h = F.relu(h)
            h = self.conv6(self.g, h)
            in_feat=torch.cat((h, torch.squeeze(exteraVar1)), 1).float()
        return h
        

loss = nn.MSELoss()
g = g.to(device)
model = UnetGraphSAGE(g,7,256, 2,num_step).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
model.train()

all_indices=np.random.permutation(np.arange(start=0, stop=int(TotalSamples/Chuncksize)))


for epoch in range(1, epochs + 1):
    
    for ss in all_indices:

        Z500train=state_training_data[variableList[0]].isel(time=slice((ss*Chuncksize),(ss+1)*Chuncksize)).coarsen(grid_yt=coarsenInd).mean().coarsen(grid_xt=coarsenInd).mean()
        T2mtrain1=state_training_data[variableList[1]].isel(time=slice((ss*Chuncksize),(ss+1)*Chuncksize)).coarsen(grid_yt=coarsenInd).mean().coarsen(grid_xt=coarsenInd).mean()
        T2mtrain2=state_training_data[variableList[2]].isel(time=slice((ss*Chuncksize),(ss+1)*Chuncksize)).coarsen(grid_yt=coarsenInd).mean().coarsen(grid_xt=coarsenInd).mean()

        Z500train=np.swapaxes(Z500train.values, 1, 0)
        T2mtrain1=np.swapaxes(T2mtrain1.values, 1, 0)
        T2mtrain2=np.swapaxes(T2mtrain2.values, 1, 0)

        T2mtrain=T2mtrain1-T2mtrain2

        T2mtrain=T2mtrain.reshape(np.size(T2mtrain, 0), np.size(T2mtrain, 1)*np.size(T2mtrain, 2)*np.size(T2mtrain, 3))
        Z500train=Z500train.reshape(np.size(Z500train, 0), np.size(Z500train, 1)*np.size(Z500train, 2)*np.size(Z500train, 3))

        # Zmean = np.mean(Z500train)
        # Zstd = np.std(Z500train)

        # Tmean = np.mean(T2mtrain)
        # Tstd = np.std(T2mtrain)


        T2mtrain=(T2mtrain-Tmean)/Tstd
        Z500train=(Z500train-Zmean)/Zstd

        

        T2mtrain=np.expand_dims(T2mtrain,axis=0)
        Z500train=np.expand_dims(Z500train,axis=0)

        dataSets=np.concatenate((Z500train,T2mtrain),axis=0)

        num_samples=np.size(dataSets,1)
        print(f"Total samples: {num_samples}")


        len_val = round(num_samples * 0.25)
        len_train = round(num_samples * 0.75)
        train = dataSets[:,: len_train]
        val = dataSets[:,len_train+14: len_train + len_val]

        x_train=train[:,0:-lead,:]
        y_train=train[:,lead::,:]
        x_val=val[:,0:-lead,:]
        y_val=val[:,lead::,:]



        x_train=np.swapaxes(x_train, 1, 0)
        y_train=np.swapaxes(y_train, 1, 0)
        x_train=np.swapaxes(x_train, 2, 1)
        y_train=np.swapaxes(y_train, 2, 1)
        x_train=torch.Tensor(x_train).to(device)
        y_train=torch.Tensor(y_train).to(device)

        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)



        x_val=np.swapaxes(x_val, 1, 0)
        y_val=np.swapaxes(y_val, 1, 0)
        x_val=np.swapaxes(x_val, 2, 1)
        y_val=np.swapaxes(y_val, 2, 1)
        x_val=torch.Tensor(x_val).to(device)
        y_val=torch.Tensor(y_val).to(device)


        val_data = torch.utils.data.TensorDataset(x_val, y_val)
        val_iter = torch.utils.data.DataLoader(val_data, batch_size)
        

        if valInde==0:
            min_val_loss = np.inf
            valInde+=1

        l_sum, n = 0.0, 0
        for x, y in train_iter:
            exteraVar1=exteraVar[:x.size(0)]
            x=torch.squeeze(torch.cat((x, exteraVar1), 2)).float() 
            y_pred = model(x,exteraVar1).view(-1 ,out_feat)
            l = loss(y_pred, torch.squeeze(y))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        
        print(" epoch", epoch, ", train loss:", l.item())
        scheduler.step()
        val_loss = evaluate_model(model, loss, val_iter,exteraVar,out_feat)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), savemodelpath)

        print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)

        # fs = get_fs(model_out_url)
        # fs.put(savemodelpath, model_out_url)
        # print(savemodelpath, model_out_url)



# best_model = STGCN_WAVE(channels, window, num_nodes, g, drop_prob, num_layers, device, control_str).to(device)
# best_model.load_state_dict(torch.load(savemodelpath))

# l = evaluate_model(best_model, loss, test_iter)
# MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
# print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
