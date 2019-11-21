import dask.array as da
import numpy as np
import xarray as xr
from numba import jit
from scipy.ndimage import map_coordinates

from vcm.convenience import open_dataset


@jit
def lagrangian_origin_coordinates(dx, dy, dz, u, v, w, h=1):
    """Semi-lagrangian advection term for data on (time, level, x, y) grid"""
    nt, nz, ny, nx = u.shape

    output = np.empty((u.ndim,) + u.shape)

    for t in range(nt):
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    alpha_x = -u[t, k, j, i] * h
                    alpha_y = -v[t, k, j, i] * h
                    alpha_z = -w[t, k, j, i] * h

                    dgrid_x = alpha_x / dx[j, i]
                    dgrid_y = alpha_y / dy[j, i]
                    dgrid_z = alpha_z / dz[t, k, j, i]

                    output[0, t, k, j, i] = t
                    output[1, t, k, j, i] = k + dgrid_z
                    output[2, t, k, j, i] = j + dgrid_y
                    output[3, t, k, j, i] = i + dgrid_x

    return output


@jit
def compute_dz(z):
    nt, nz, ny, nx = z.shape
    output = np.empty(z.shape)
    for t in range(nt):
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if k == 0:
                        tmp = z[t, 1, j, i] - z[t, 0, j, i]
                    elif k == nz - 1:
                        tmp = z[t, -1, j, i] - z[t, -2, j, i]
                    else:
                        tmp = (z[t, k + 1, j, i] - z[t, k - 1, j, i]) / 2

                    output[t, k, j, i] = tmp

    return output


def compute_dx_dy(lat, lon):
    # convert lat/lon to meters
    radius_earth = 6378e3

    dlon = np.r_[lon[1] - lon[0], lon[2:] - lon[:-2], lon[-1] - lon[-2]]
    dlat = np.r_[lat[1] - lat[0], lat[2:] - lat[:-2], lat[-1] - lat[-2]]

    dx = np.cos(np.deg2rad(lat[:, np.newaxis])) * np.deg2rad(dlon) * radius_earth
    dy = np.deg2rad(dlat) * radius_earth
    dy = np.broadcast_to(dy[:, np.newaxis], (len(lat), len(lon)))

    return dx, dy


def lagrangian_update(phi, *args, **kwargs):
    origin = lagrangian_origin_coordinates(*args, **kwargs)
    return map_coordinates(phi, origin)


# Dask wrapper

ghost_cells_depth = {1: 2, 2: 2, 3: 2}
boundaries = {1: "nearest", 2: "reflect", 3: "nearest"}


def ghost_2d(x):
    return da.overlap.overlap(
        x, depth={0: 2, 1: 2}, boundary={0: "nearest", 1: "nearest"}
    )


def ghosted_array(x, time_axis=0):
    depth = {key + time_axis: val for key, val in ghost_cells_depth.items()}
    boundary = {key + time_axis: val for key, val in boundaries.items()}
    return da.overlap.overlap(x, depth=depth, boundary=boundary)


def remove_ghost_cells(x, time_axis=0):
    depth = {key + time_axis: val for key, val in ghost_cells_depth.items()}
    return da.overlap.trim_internal(x, depth)


def lagrangian_update_dask(phi, lat, lon, dz, u, v, w, **kwargs):
    """Compute semi-lagrangian update of a variable

    :param phi: dask
    :param lat: numpy
    :param lon: numpy
    :param dz: dask
    :param u: dask
    :param v: dask
    :param w: dask
    :return: dask
    """
    dx, dy = compute_dx_dy(np.asarray(lat), np.asarray(lon))
    horz_chunks = u.chunks[-2:]
    ghosted_2d = [ghost_2d(da.from_array(arg, horz_chunks)) for arg in [dx, dy]]
    ghosted_3d = [ghosted_array(arg) for arg in [dz, u, v, w]]
    ghosted_phi = ghosted_array(phi)

    # construct argument list for blockwise
    arg_inds_lagrangian_update = ["tkji"] + ["ji"] * 2 + ["tkji"] * 4
    ghosted_args = [ghosted_phi] + ghosted_2d + ghosted_3d

    blockwise_args = []
    for arr, ind in zip(ghosted_args, arg_inds_lagrangian_update):
        blockwise_args.append(arr)
        blockwise_args.append(ind)

    # call blockwise
    output_ghosted = da.blockwise(
        lagrangian_update, "tkji", *blockwise_args, dtype=ghosted_phi.dtype, **kwargs
    )
    return remove_ghost_cells(output_ghosted)


def lagrangian_update_xarray(data_3d, advect_var="temp", **kwargs):
    dim_order = ["time", "pfull", "y", "x"]
    ds = data_3d.assign(dz=data_3d.dz.transpose(*dim_order))
    args = [ds[key].data for key in [advect_var, "y", "x", "dz", "u", "v", "w"]]
    dask = lagrangian_update_dask(*args, **kwargs)
    return xr.DataArray(dask, coords=ds[advect_var].coords, dims=ds[advect_var].dims)


def advection_fixed_height(
    data_3d: xr.Dataset, advect_var: str = "temp", h: float = 30.0
) -> xr.DataArray:
    """Compute the advection terms at a fixed heights

    This estimates the advection terms

        v.grad(phi)

    using a semi-Lagrangian scheme.

    Note:
      Add to the output of storage_fixed_height to compute the total derivative
    """
    advected = lagrangian_update_xarray(data_3d, advect_var, h=h)
    return (advected - data_3d[advect_var]) / h


def reverse_dim(ds, dim="pfull"):
    return ds.isel({dim: slice(None, None, -1)})


def height_interfaces(dz: xr.DataArray, zs: xr.DataArray = 0) -> xr.DataArray:
    """Compute the height from the surface elevation and thickness

    Args:
        dz: thickness of layers. assumes that top of atmosphere is index 0 in the pfull dimension
        zs: surface elevation

    Returns:
        height: height of the vertical levels

    """
    # TODO: Unused, probably remove -- AP
    # pfull = dz["pfull"]

    dz = dz.drop("pfull")
    zero = xr.zeros_like(dz.isel(pfull=slice(0, 1)))
    dz = xr.concat([dz, zero], dim="pfull")
    zint = reverse_dim(reverse_dim(dz).cumsum("pfull")) + zs

    return zint.rename({"pfull": "pfull_b"})


def vertical_interfaces_to_centers(phi):
    try:
        phi = phi.drop("pfull_b")
    except ValueError:
        pass
    avg = (phi.isel(pfull_b=slice(1, None)) + phi.isel(pfull_b=slice(0, -1))) / 2
    return avg.rename({"pfull_b": "pfull"})


def height_centered(dz: xr.DataArray, zs: xr.DataArray = 0) -> xr.DataArray:
    zint = height_interfaces(dz, zs)
    z = vertical_interfaces_to_centers(zint)
    # these functions drop the pfull coordinate
    return z.assign_coords(pfull=dz.pfull)


def storage_fixed_height(
    phi: xr.DataArray, z_centered, dz: xr.DataArray, dt: float = 3 * 3600
) -> xr.DataArray:
    """Compute the storage at fixed height using the chain rule

    (f_t)_z = (f_t)_sigma - f_z (z_t)_sigma

    where the second sub-script denotes constant sigma or height.

    """
    pfull = phi["pfull"]
    phi = phi.drop("pfull")
    dz = dz.drop("pfull")
    z = z_centered.drop("pfull")

    dphi_dz = phi.differentiate("pfull") / dz
    dphi_dt = (phi.shift(time=-1) - phi) / dt
    dz_dt = (z.shift(time=-1) - z) / dt
    ans = dphi_dt - dphi_dz * dz_dt
    return ans.assign_coords(pfull=pfull)


def average_end_points(phi: xr.DataArray):
    return (phi.shift(time=-1) + phi) / 2


def apparent_source(scalar, z_centered, dz, advection_tendency):
    return storage_fixed_height(scalar, z_centered, dz) - average_end_points(
        advection_tendency
    )


def storage_and_advection(data_3d, tracers, time_step):
    data_vars = {}
    z_c = height_centered(data_3d["dz"], data_3d["zs"])
    for key in tracers:
        data_vars["advection_" + key] = advection_fixed_height(data_3d, key)
        storage = storage_fixed_height(data_3d[key], z_c, data_3d.dz, dt=time_step)
        data_vars["storage_" + key] = storage

    return xr.Dataset(data_vars)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_key")
    parser.add_argument("output_zarr")
    parser.add_argument("--time-step", type=float, default=3 * 3600)

    advect_variables = ["qv", "temp"]

    args = parser.parse_args()

    data_3d = open_dataset(args.input_key)
    print("Processing this 3D dataset:")
    print()
    print(data_3d)
    sources = storage_and_advection(
        data_3d, tracers=advect_variables, time_step=args.time_step
    )
    sources.to_zarr(args.output_zarr, mode="a")


if __name__ == "__main__":
    main()
