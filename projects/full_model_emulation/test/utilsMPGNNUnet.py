import torch
import numpy as np


def evaluate_model(
    model, loss, data_iter, exteraVar, out_feat, latlon3, latlon4, latlon5, device
):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            exteraVar1 = exteraVar[: x.size(0)]
            x = torch.squeeze(torch.cat((x.to(device), exteraVar1), 2)).float()
            y_pred = model(x, latlon3, latlon4, latlon5).view(-1, out_feat)
            l = loss(y_pred, torch.squeeze(y.to(device)))
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def evaluate_model11(
    model, loss, data_iter, exteraVar, out_feat,latlon1, latlon2,latlon3, latlon4, latlon5, device
):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            exteraVar1 = exteraVar[: x.size(0)]
            x = torch.squeeze(torch.cat((x.to(device), exteraVar1), 2)).float()
            y_pred = model(x, latlon1, latlon2,latlon3, latlon4, latlon5).view(-1, out_feat)
            l = loss(y_pred, torch.squeeze(y.to(device)))
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def evaluate_model2(
    model, loss, data_iter, exteraVar, out_feat, device):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            exteraVar1 = exteraVar[: x.size(0)]
            x = torch.squeeze(torch.cat((x.to(device), exteraVar1), 2)).float()
            y_pred = model(x,exteraVar1).view(-1, out_feat)
            l = loss(y_pred, torch.squeeze(y.to(device)))
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def evaluate_model3(
    model, loss, data_iter, exteraVar, out_feat, etype1, etype2, etype3, etype4, etype5, device):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            exteraVar1 = exteraVar[: x.size(0)]
            x = torch.squeeze(torch.cat((x.to(device), exteraVar1), 2)).float()
            y_pred = model(x, etype1, etype2, etype3, etype4, etype5).view(-1, out_feat)
            l = loss(y_pred, torch.squeeze(y.to(device)))
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def evaluate_model4(
    model, loss, data_iter, exteraVar, out_feat, etype1, etype2, etype3 , device):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            exteraVar1 = exteraVar[: x.size(0)]
            x = torch.squeeze(torch.cat((x.to(device), exteraVar1), 2)).float()
            y_pred = model(x, etype1, etype2, etype3).view(-1, out_feat)
            l = loss(y_pred, torch.squeeze(y.to(device)))
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(
                model(x).view(len(x), -1).cpu().numpy()
            ).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, MAPE, RMSE
