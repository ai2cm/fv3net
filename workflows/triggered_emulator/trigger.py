import xarray as xr
import numpy as np


def is_active(a):
    threshold_tendency = 0.01 / 86400  # K/day
    return ((a.dQ1 ** 2 + (2.51e6 / 1004 * a.dQ2) ** 2) > threshold_tendency ** 2).any(
        "z"
    )


def resample_inactive(x, is_active=is_active):
    x['active'] = is_active(x)
    active = x['active'].values

    (i_active,) = np.nonzero(active)
    (i_inactive,) = np.nonzero(~active)

    assert len(i_active) < len(i_inactive), len(i_active)

    inactive_sampled = np.random.choice(i_inactive, size=len(i_active), replace=False)
    data = xr.concat(
        [x.isel(sample=i_active), x.isel(sample=inactive_sampled)], dim="sample"
    )

    return data