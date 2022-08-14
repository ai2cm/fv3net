import torch
import numpy as np


def evaluate_model(model, latlon, loss, data_iter, exteraVar, out_feat, device):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y = y.to(device)
            x = x.to(device)
            exteraVar1 = exteraVar[: x.size(0)]
            # x = torch.cat((x, exteraVar1), 1).float()
            x = torch.squeeze(torch.cat((x.to(device), exteraVar1), 2)).float()
            y_pred = model(x,latlon).view(-1, out_feat)
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