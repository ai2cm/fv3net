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
import wandb
from fv3net.artifacts.resolve_url import resolve_url
from vcm import get_fs
from SAGE_Unet_MP_LowRes import UnetGraphSAGE

halo=1
control_str = "Halo_SAGEUnet"  #'TNSTTNST' #'TNTSTNTST'


lead = 6
residual = 0
coarsenInd = 1
n_filter = 256
edge_h_feats = 64
input_res = 48
pooling_size = 2
edge_in_feats = 2
num_step_message_passing = 5
edgenormal=0

if halo==1:
    print("halo")
    g1 = pickle.load(open("NewHalo_Graph5_Coarsen48", "rb"))

    g2 = pickle.load(open("NewHalo_Graph5_Coarsen24", "rb"))

    g3 = pickle.load(open("NewHalo_Graph5_Coarsen12", "rb"))
    coarsenInd3 = 4

    g4 = pickle.load(open("NewHalo_Graph5_Coarsen6", "rb"))
    coarsenInd4 = 8

    g5 = pickle.load(open("NewHalo_Graph5_Coarsen3", "rb"))
    coarsenInd5 = 16

elif halo==0:
    print("No halo")
    g1 = pickle.load(open("UpdatedGraph_Neighbour5_Coarsen1", "rb"))

    g2 = pickle.load(open("UpdatedGraph_Neighbour5_Coarsen2", "rb"))

    g3 = pickle.load(open("UpdatedGraph_Neighbour5_Coarsen4", "rb"))
    coarsenInd3 = 4

    g4 = pickle.load(open("UpdatedGraph_Neighbour5_Coarsen8", "rb"))
    coarsenInd4 = 8

    g5 = pickle.load(open("UpdatedGraph_Neighbour5_Coarsen16", "rb"))
    coarsenInd5 = 16

# g1=build_graph(48)
# g2=build_graph(24)
# g3=build_graph(12)
# g4=build_graph(6)
# g5=build_graph(3)


print(control_str)

epochs = 30

variableList = ["h500", "h200", "h850"]
TotalSamples = 8500
Chuncksize = 2000


lr = 0.001
disablecuda = "store_true"
batch_size = 1
drop_prob = 0
out_feat = 2

savemodelpath = (
    "No_diff_latlon_LowRes_Shift_Ave_latlon_Unet_MP_All5_edges_Orininal_New_Pooling_weight_layer_"
    + control_str
    +"edgeNormal_"
    +str(edgenormal)
    + "Poolin"
    + "Meanpool"
    + "hidden_filetrs"
    + str(n_filter)
    +"edgeHidden"
    +str(edge_h_feats)
    + "learning_rate"
    + str(lr)
    + "_lead"
    + str(lead)
    + "_epochs_"
    + str(epochs)
    + "MP_Block_"
    + str(num_step_message_passing)
    + "coarsen_"
    + str(coarsenInd)
    + "residual_"
    + str(residual)
    + ".pt"
)

print(savemodelpath)

BUCKET = "vcm-ml-experiments"
PROJECT = "full-model-emulation"

model_out_url = resolve_url(BUCKET, PROJECT, savemodelpath)
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


lat1 = lat_lon_data.latitude[1].load()
lon1 = lat_lon_data.longitude[1].load()
lat = lat1.values.flatten()
lon = lon1.values.flatten()

lat2=np.zeros([6,24,24])
lon2=np.zeros([6,24,24])

lat3=np.zeros([6,12,12])
lon3=np.zeros([6,12,12])

lat4=np.zeros([6,6,6])
lon4=np.zeros([6,6,6])

lat5=np.zeros([6,3,3])
lon5=np.zeros([6,3,3])

for i in range(6):
    lat2[i]=0.25* (lat1[i,1::2, :-1:2] + lat1[i,:-1:2, :-1:2] + lat1[i,1::2, 1::2] + lat1[i,:-1:2, :-1:2])
    lon2[i]=0.25* (lon1[i,1::2, :-1:2] + lon1[i,:-1:2, :-1:2] + lon1[i,1::2, 1::2] + lon1[i,:-1:2, :-1:2])

for i in range(6):
    lat3[i]=0.25* (lat2[i,1::2, :-1:2] + lat2[i,:-1:2, :-1:2] + lat2[i,1::2, 1::2] + lat2[i,:-1:2, :-1:2])
    lon3[i]=0.25* (lon2[i,1::2, :-1:2] + lon2[i,:-1:2, :-1:2] + lon2[i,1::2, 1::2] + lon2[i,:-1:2, :-1:2])

for i in range(6):
    lat4[i]=0.25* (lat3[i,1::2, :-1:2] + lat3[i,:-1:2, :-1:2] + lat3[i,1::2, 1::2] + lat3[i,:-1:2, :-1:2])
    lon4[i]=0.25* (lon3[i,1::2, :-1:2] + lon3[i,:-1:2, :-1:2] + lon3[i,1::2, 1::2] + lon3[i,:-1:2, :-1:2])

for i in range(6):
    lat5[i]=0.25* (lat4[i,1::2, :-1:2] + lat4[i,:-1:2, :-1:2] + lat4[i,1::2, 1::2] + lat4[i,:-1:2, :-1:2])
    lon5[i]=0.25* (lon4[i,1::2, :-1:2] + lon4[i,:-1:2, :-1:2] + lon4[i,1::2, 1::2] + lon4[i,:-1:2, :-1:2])

lat2=lat2.flatten()
lat3=lat3.flatten()
lat4=lat4.flatten()
lat5=lat5.flatten()

lon2=lon2.flatten()
lon3=lon3.flatten()
lon4=lon4.flatten()
lon5=lon5.flatten()


cosLat = np.cos(lat)
cosLon = np.cos(lon)
sinLat = np.sin(lat)
sinLon = np.sin(lon)
for i in range(2):
    if i == 0:
        sinLon = torch.tensor(sinLon).unsqueeze(0).repeat(1, 1)
        cosLon = torch.tensor(cosLon).unsqueeze(0).repeat(1, 1)
        sinLat = torch.tensor(sinLat).unsqueeze(0).repeat(1, 1)
        cosLat = torch.tensor(cosLat).unsqueeze(0).repeat(1, 1)
        landSea_Mask = torch.tensor(landSea_Mask).unsqueeze(0).repeat(1, 1)
    elif i == 1:
        sinLon = (sinLon).unsqueeze(0).repeat(batch_size, 1, 1)
        cosLon = (cosLon).unsqueeze(0).repeat(batch_size, 1, 1)
        sinLat = (sinLat).unsqueeze(0).repeat(batch_size, 1, 1)
        cosLat = (cosLat).unsqueeze(0).repeat(batch_size, 1, 1)
        landSea_Mask = (landSea_Mask).unsqueeze(0).repeat(batch_size, 1, 1)

exteraVar = torch.cat((sinLon, sinLat, cosLon, cosLat, landSea_Mask), 1).to(device)
exteraVar = np.swapaxes(exteraVar, 2, 1)
print(device)

num_nodes = len(lon)
print(f"numebr of grids: {num_nodes}")


edg = np.asarray(g1.edges())
latInd = lat[edg[1]]
lonInd = lon[edg[1]]

# latInd1 = lat[edg[0]]
# lonInd1 = lon[edg[0]]

# latInd = latInd - latInd1
# lonInd = lonInd - lonInd1

if edgenormal==1:
    latInd=(latInd-np.mean(latInd))/np.std(latInd)
    lonInd=(lonInd-np.mean(lonInd))/np.std(lonInd)

latlon1 = [latInd.T, lonInd.T]
latlon1 = torch.from_numpy(np.swapaxes(latlon1, 1, 0)).float()
latlon1 = latlon1.to(device)

del edg, lonInd, latInd, lonInd1, latInd1

edg = np.asarray(g2.edges())
latInd = lat2[edg[1]]
lonInd = lon2[edg[1]]

# latInd1 = lat2[edg[0]]
# lonInd1 = lon2[edg[0]]

# latInd = latInd - latInd1
# lonInd = lonInd - lonInd1

if edgenormal==1:
    latInd=(latInd-np.mean(latInd))/np.std(latInd)
    lonInd=(lonInd-np.mean(lonInd))/np.std(lonInd)

latlon2 = [latInd.T, lonInd.T]
latlon2 = torch.from_numpy(np.swapaxes(latlon2, 1, 0)).float()
latlon2 = latlon2.to(device)

del edg, lonInd, latInd, lonInd1, latInd1


edg = np.asarray(g3.edges())
latInd = lat3[edg[1]]
lonInd = lon3[edg[1]]

# latInd1 = lat3[edg[0]]
# lonInd1 = lon3[edg[0]]

# latInd = latInd - latInd1
# lonInd = lonInd - lonInd1

if edgenormal==1:
    latInd=(latInd-np.mean(latInd))/np.std(latInd)
    lonInd=(lonInd-np.mean(lonInd))/np.std(lonInd)

latlon3 = [latInd.T, lonInd.T]
latlon3 = torch.from_numpy(np.swapaxes(latlon3, 1, 0)).float()
latlon3 = latlon3.to(device)

del edg, lonInd, latInd, lonInd1, latInd1

edg = np.asarray(g4.edges())
latInd = lat4[edg[1]]
lonInd = lon4[edg[1]]

# latInd1 = lat4[edg[0]]
# lonInd1 = lon4[edg[0]]

# latInd = latInd - latInd1
# lonInd = lonInd - lonInd1

if edgenormal==1:
    latInd=(latInd-np.mean(latInd))/np.std(latInd)
    lonInd=(lonInd-np.mean(lonInd))/np.std(lonInd)

latlon4 = [latInd.T, lonInd.T]
latlon4 = torch.from_numpy(np.swapaxes(latlon4, 1, 0)).float()
latlon4 = latlon4.to(device)

del edg, lonInd, latInd, lonInd1, latInd1

edg = np.asarray(g5.edges())
latInd = lat5[edg[1]]
lonInd = lon5[edg[1]]

# latInd1 = lat5[edg[0]]
# lonInd1 = lon5[edg[0]]

# latInd = latInd - latInd1
# lonInd = lonInd - lonInd1

if edgenormal==1:
    latInd=(latInd-np.mean(latInd))/np.std(latInd)
    lonInd=(lonInd-np.mean(lonInd))/np.std(lonInd)

latlon5 = [latInd.T, lonInd.T]
latlon5 = torch.from_numpy(np.swapaxes(latlon5, 1, 0)).float()
latlon5 = latlon5.to(device)
del edg, lonInd, latInd, lonInd1, latInd1


Zmean = 5765.8457  # Z500mean=5765.8457,
Zstd = 90.79599  # Z500std=90.79599

Tmean = 10643.382  # Thickmean=10643.382
Tstd = 162.12427  # Thickstd=162.12427
valInde = 0

print("loading model")

loss = nn.MSELoss()
g1 = g1.to(device)
g2 = g2.to(device)
g3 = g3.to(device)
g4 = g4.to(device)
g5 = g5.to(device)

model = UnetGraphSAGE(
    input_res,
    pooling_size,
    g1,
    g2,
    g3,
    g4,
    g5,
    7,
    n_filter,
    2,
    edge_in_feats,
    edge_h_feats,
    num_step_message_passing,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
print('Total Parameters:', sum([p.nelement() for p in model.parameters()]))


for epoch in range(1, epochs + 1):
    all_indices = np.random.permutation(
        np.arange(start=0, stop=int(TotalSamples / Chuncksize))
    )

    for ss in all_indices:
        model.train()
        Z500train = (
            state_training_data[variableList[0]]
            .isel(time=slice((ss * Chuncksize), (ss + 1) * Chuncksize))
            .coarsen(grid_yt=coarsenInd)
            .mean()
            .coarsen(grid_xt=coarsenInd)
            .mean()
        )
        T2mtrain1 = (
            state_training_data[variableList[1]]
            .isel(time=slice((ss * Chuncksize), (ss + 1) * Chuncksize))
            .coarsen(grid_yt=coarsenInd)
            .mean()
            .coarsen(grid_xt=coarsenInd)
            .mean()
        )
        T2mtrain2 = (
            state_training_data[variableList[2]]
            .isel(time=slice((ss * Chuncksize), (ss + 1) * Chuncksize))
            .coarsen(grid_yt=coarsenInd)
            .mean()
            .coarsen(grid_xt=coarsenInd)
            .mean()
        )

        Z500train = np.swapaxes(Z500train.values, 1, 0)
        T2mtrain1 = np.swapaxes(T2mtrain1.values, 1, 0)
        T2mtrain2 = np.swapaxes(T2mtrain2.values, 1, 0)

        T2mtrain = T2mtrain1 - T2mtrain2

        T2mtrain = T2mtrain.reshape(
            np.size(T2mtrain, 0),
            np.size(T2mtrain, 1) * np.size(T2mtrain, 2) * np.size(T2mtrain, 3),
        )
        Z500train = Z500train.reshape(
            np.size(Z500train, 0),
            np.size(Z500train, 1) * np.size(Z500train, 2) * np.size(Z500train, 3),
        )

        # Zmean = np.mean(Z500train)
        # Zstd = np.std(Z500train)

        # Tmean = np.mean(T2mtrain)
        # Tstd = np.std(T2mtrain)

        T2mtrain = (T2mtrain - Tmean) / Tstd
        Z500train = (Z500train - Zmean) / Zstd

        T2mtrain = np.expand_dims(T2mtrain, axis=0)
        Z500train = np.expand_dims(Z500train, axis=0)

        dataSets = np.concatenate((Z500train, T2mtrain), axis=0)

        num_samples = np.size(dataSets, 1)
        print(f"Total samples: {num_samples}")

        len_val = round(num_samples * 0.25)
        len_train = round(num_samples * 0.75)
        train = dataSets[:, :len_train]
        val = dataSets[:, len_train + 14 : len_train + len_val]

        x_train = train[:, 0:-lead, :]
        y_train = train[:, lead::, :]
        x_val = val[:, 0:-lead, :]
        y_val = val[:, lead::, :]

        if residual == 1:
            y_train = y_train - x_train
            y_val = y_val - x_val

        x_train = np.swapaxes(x_train, 1, 0)
        y_train = np.swapaxes(y_train, 1, 0)
        x_train = np.swapaxes(x_train, 2, 1)
        y_train = np.swapaxes(y_train, 2, 1)
        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)

        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)

        x_val = np.swapaxes(x_val, 1, 0)
        y_val = np.swapaxes(y_val, 1, 0)
        x_val = np.swapaxes(x_val, 2, 1)
        y_val = np.swapaxes(y_val, 2, 1)
        x_val = torch.Tensor(x_val)
        y_val = torch.Tensor(y_val)

        val_data = torch.utils.data.TensorDataset(x_val, y_val)
        val_iter = torch.utils.data.DataLoader(val_data, batch_size)

        if valInde == 0:
            min_val_loss = np.inf
            valInde += 1

        l_sum, n = 0.0, 0
        for x, y in train_iter:
            ss=time.time()
            exteraVar1 = exteraVar[: x.size(0)]
            x = torch.squeeze(torch.cat((x.to(device), exteraVar1), 2)).float()
            optimizer.zero_grad()
            y_pred = model(x, latlon1,latlon2,latlon3, latlon4, latlon5).view(-1, out_feat)
            l = loss(y_pred, torch.squeeze(y.to(device)))
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        print(" epoch", epoch, ", train loss:", l.item(),"time:",time.time()-ss)
        scheduler.step()
        val_loss = evaluate_model11(
            model,
            loss,
            val_iter,
            exteraVar,
            out_feat,
            latlon1,
            latlon2,
            latlon3,
            latlon4,
            latlon5,
            device,
        )
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), savemodelpath)
        print(
            "epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss
        )

        fs = get_fs(model_out_url)
        fs.put(savemodelpath, model_out_url)
        print(savemodelpath, model_out_url)


# best_model = STGCN_WAVE(channels, window, num_nodes, g, drop_prob, num_layers, device, control_str).to(device)
# best_model.load_state_dict(torch.load(savemodelpath))

# l = evaluate_model(best_model, loss, test_iter)
# MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
# print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
