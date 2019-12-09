import argparse

import numpy as np
import xarray as xr

from vcm import cubedsphere

gravity = 9.81  # m /s2
Rd = 287  # J / K / kg

# default for restart file
VERTICAL_DIM = "zaxis_1"


def density_to_virtual_temperature(rho, p):
    return p / Rd / rho


def virtual_temperature_to_temperature(tv, qv):
    return tv / (1 + 0.61 * qv)


def density_from_thickness(dz, dp):
    return np.abs(dp) / np.abs(dz) / gravity


def absolute_pressure_interface(dp, toa_pressure=300, dim=VERTICAL_DIM):
    dpv = dp.variable
    top = 0 * dpv.isel({dim: [0]}) + toa_pressure
    dp_with_top = top.concat([top, dpv], dim=dim)
    return dp_with_top.cumsum(dim)


def interface_to_center(ds, dim=VERTICAL_DIM):
    return (ds.isel({dim: slice(0, -1)}) + ds.isel({dim: slice(1, None)})) / 2


def dp_to_p(dp, dim=VERTICAL_DIM):
    pi = absolute_pressure_interface(dp, dim=dim)
    pc = interface_to_center(pi)
    return xr.DataArray(pc, coords=dp.coords)


def hydrostatic_temperature(dz, dp, p, q):

    assert q.dims == dp.dims, q.dims
    rho = density_from_thickness(dz, dp)
    tv = density_to_virtual_temperature(rho, p)
    return virtual_temperature_to_temperature(tv, q)


def hydrostatic_temperature_with_logp(dz, dp, q):
    pi = absolute_pressure_interface(dp)
    dlogp = xr.DataArray(np.log(pi)).diff(VERTICAL_DIM)
    tv = gravity * np.abs(dz) / np.abs(dlogp) / Rd
    return virtual_temperature_to_temperature(tv, q)


def rho_from_temp(t, q, p):
    return p / (Rd * t * (1 + 0.61 * q))


def rho_from_dz(dz, dp):
    return np.abs(dp) / np.abs(dz) / gravity


def hydrostatic_dz(T, q, dp):
    p = dp_to_p(dp)
    tv = (1 + 0.61 * q) * T
    dlogp = dp / p
    return -dlogp * Rd * tv / gravity


def hydrostatic_dz_withlog(T, q, dp):
    pi = absolute_pressure_interface(dp)

    tv = (1 + 0.61 * q) * T
    dlogp = xr.DataArray(np.log(pi)).diff(VERTICAL_DIM)
    return -dlogp * Rd * tv / gravity


def thickness(x, p):
    return x.where((p < 600e2) & (p > 100e2)).sum(VERTICAL_DIM)


# TODO: refactor this to a separate file for adjustments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Hydrostatically adjust the height variable"
    )
    parser.add_argument("tracer", help="everything before tile.nc")
    parser.add_argument("fv_core", help="everything before tile.nc")
    parser.add_argument("fv_core_out", help="output prefix")
    parser.add_argument(
        "--adjust-temp",
        action="store_true",
        help="adjust the temperature rather than the thickness",
    )
    return parser.parse_args()


def open_tiles(prefix):
    return xr.open_mfdataset(prefix + ".tile?.nc", combine="nested", concat_dim="tile")


def main():
    args = parse_arguments()
    tracer = open_tiles(args.tracer)
    core = open_tiles(args.fv_core)

    # rename q from tracer file to match the cell-centered dims of the
    # fv_core_file
    tracer_rename_dict = dict(zip(tracer.sphum.dims, core.delp.dims))
    sphum = tracer.sphum.rename(tracer_rename_dict)

    if args.adjust_temp:
        print("Adjusting temperature")
        new_temp = hydrostatic_temperature_with_logp(core.DZ, core.delp, sphum)
        core_adj = core.assign(T=new_temp)
    else:
        print("Adjusting height")
        dz_adjusted = hydrostatic_dz_withlog(core.T, sphum, core.delp)
        core_adj = core.assign(DZ=dz_adjusted)

    # save to disk
    cubedsphere.save_tiles_separately(core_adj, args.fv_core_out, output_directory=".")


if __name__ == "__main__":
    main()
