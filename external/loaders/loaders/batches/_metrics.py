def _bias(da_target, da_pred, weights=None, mean_dims=None):
    bias = da_pred - da_target
    if weights is not None:
        bias *= weights
    return bias.mean(dim=mean_dims)


def _rmse(da_target, da_pred, weights=None, mean_dims=None):
    se = (da_target - da_pred) ** 2
    if weights is not None:
        se *= weights
    return np.sqrt(se.mean(dim=mean_dims))