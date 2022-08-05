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

multiStep = 8
lead = 6
hidden_filetrs = 64
coarsenInd = 3
g = pickle.load(open("UpdatedGraph_Neighbour10_Coarsen3", "rb"))
residual = 0


control_str = "SAGEUnet_Inverse"  #'TNSTTNST' #'TNTSTNTST'

print(control_str)

epochs = 30

variableList = ["h500", "h200", "h850"]
TotalSamples = 8500
Chuncksize = 1000
num_step = 1
aggregat = "gcn"


lr = 0.001
disablecuda = "store_true"
batch_size = 1
drop_prob = 0
out_feat = 2

savemodelpath = (
    "weight_layer_"
    + control_str
    + "hidden_filetrs"
    + str(hidden_filetrs)
    + "_MultistepLoss"
    + str(multiStep)
    + "_lead"
    + str(lead)
    + "_epochs_"
    + str(epochs)
    + "MP_Block_"
    + str(num_step)
    + "aggregat_"
    + aggregat
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


lat = lat_lon_data.latitude[1].load()
lon = lat_lon_data.longitude[1].load()
lat = lat[:, ::coarsenInd, ::coarsenInd].values.flatten()
lon = lon[:, ::coarsenInd, ::coarsenInd].values.flatten()
# cosLat=np.expand_dims(np.cos(lat),axis=1)
# cosLon=np.expand_dims(np.cos(lon),axis=1)
# sinLat=np.expand_dims(np.sin(lat),axis=1)
# sinLon=np.expand_dims(np.sin(lon),axis=1)
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


edg = np.asarray(g.edges())
latInd = lat[edg[1]]
lonInd = lon[edg[1]]
latlon = [latInd.T, lonInd.T]
# latlon=np.swapaxes(latlon, 1, 0)
latlon = torch.from_numpy(np.swapaxes(latlon, 1, 0)).float()
latlon = latlon.to(device)


Zmean = 5765.8457  # Z500mean=5765.8457,
Zstd = 90.79599  # Z500std=90.79599

Tmean = 10643.382  # Thickmean=10643.382
Tstd = 162.12427  # Thickstd=162.12427
valInde = 0

print("loading model")


class UnetGraphSAGE(nn.Module):
    def __init__(self, g, in_feats, h_feats, out_feat, num_step, aggregat):
        super(UnetGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, int(h_feats / 16), aggregat)
        self.conv2 = SAGEConv(int(h_feats / 16), int(h_feats / 16), aggregat)
        self.conv3 = SAGEConv(int(h_feats / 16), int(h_feats / 8), aggregat)
        self.conv4 = SAGEConv(int(h_feats / 8), int(h_feats / 4), aggregat)
        self.conv5 = SAGEConv(int(h_feats / 4), int(h_feats / 2), aggregat)
        self.conv6 = SAGEConv(int(h_feats / 2), int(h_feats), aggregat)
        self.conv7 = SAGEConv(int(h_feats), int(h_feats), aggregat)
        self.conv8 = SAGEConv(int(h_feats), int(h_feats / 2), aggregat)
        self.conv9 = SAGEConv(int(h_feats / 2), int(h_feats / 4), aggregat)
        self.conv10 = SAGEConv(int(h_feats / 4), int(h_feats / 8), aggregat)
        self.conv11 = SAGEConv(int(h_feats / 8), int(h_feats / 16), aggregat)
        self.conv12 = SAGEConv(int(h_feats / 16), out_feat, aggregat)
        self.g = g
        self.num_step = num_step

    def forward(self, in_feat, exteraVar1):

        for _ in range(self.num_step):
            h1 = F.relu(self.conv1(self.g, in_feat))
            h2 = F.relu(self.conv2(self.g, h1))
            h3 = F.relu(self.conv3(self.g, h2))
            h4 = F.relu(self.conv4(self.g, h3))
            h5 = F.relu(self.conv5(self.g, h4))
            h6 = F.relu(self.conv6(self.g, h5))
            h7 = F.relu(self.conv7(self.g, h6))
            h8 = torch.cat((F.relu(self.conv8(self.g, h7)), h5), dim=1)
            h8 = F.relu(self.conv8(self.g, h8))
            h9 = torch.cat((F.relu(self.conv9(self.g, h8)), h4), dim=1)
            h9 = F.relu(self.conv9(self.g, h9))
            h10 = torch.cat((F.relu(self.conv10(self.g, h9)), h3), dim=1)
            h10 = F.relu(self.conv10(self.g, h10))
            h11 = torch.cat((F.relu(self.conv11(self.g, h10)), h2), dim=1)
            h11 = F.relu(self.conv11(self.g, h11))
            out = self.conv12(self.g, h11)
            in_feat = torch.cat((out, torch.squeeze(exteraVar1)), 1).float()
        return out


loss = nn.MSELoss()
g = g.to(device)
model = UnetGraphSAGE(g, 7, hidden_filetrs, 2, num_step, aggregat).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)


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

        x_train = train[:, 0 : -multiStep * lead, :]

        y_train = x_t = np.zeros(
            [
                np.size(train, 0),
                np.size(train, 1) - multiStep * lead,
                np.size(train, 2),
                multiStep,
            ]
        )
        for sm in range(1, multiStep + 1):
            if sm == multiStep:
                y_train[:, :, :, sm - 1] = train[:, (sm) * lead : :, :]
            else:
                y_train[:, :, :, sm - 1] = train[
                    :, (sm) * lead : -(multiStep - sm) * lead, :
                ]

        x_val = val[:, 0:-lead, :]
        y_val = val[:, lead::, :]

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
        x_val = torch.Tensor(x_val).to(device)
        y_val = torch.Tensor(y_val).to(device)

        val_data = torch.utils.data.TensorDataset(x_val, y_val)
        val_iter = torch.utils.data.DataLoader(val_data, batch_size)

        if valInde == 0:
            min_val_loss = np.inf
            valInde += 1

        l_sum, n = 0.0, 0
        for x, y in train_iter:
            optimizer.zero_grad()
            exteraVar1 = exteraVar[: x.size(0)]
            x = torch.squeeze(torch.cat((x.to(device), exteraVar1), 2)).float()
            y_pred = model(x, exteraVar1).view(-1, out_feat)

            l = loss(y_pred, torch.squeeze(y[:, :, :, 0]).to(device))

            for sm in range(1, multiStep):
                y_pred2 = model(
                    torch.cat((y_pred, torch.squeeze(exteraVar1)), 1).float(),
                    exteraVar1,
                ).view(-1, out_feat)
                l += loss(y_pred2, torch.squeeze(y[:, :, :, sm]).to(device))

            l = l / multiStep
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        print(" epoch", epoch, ", train loss:", l.item())
        scheduler.step()
        val_loss = evaluate_model(model, loss, val_iter, exteraVar, out_feat)
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
