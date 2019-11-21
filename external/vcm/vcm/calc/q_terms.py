"""
Compute Q1 and Q2 from advective tendencies.
"""
import datetime

import numpy as np
import xarray as xr

G = 9.81
C_P = 1004


def compute_Q_terms(data_3d, data_adv):
    assert len(data_3d.time) == len(data_adv.time), "Time dimensions must match."
    data_3d = data_3d.chunk({"time": len(data_3d.time)})
    data_adv = data_adv.chunk({"time": len(data_adv.time)})

    # Compute time tendencies
    dt = data_3d.time.diff("time").values[0] / np.timedelta64(1, "s")
    half_time = data_3d.time.values[1:] - datetime.timedelta(seconds=dt / 2)
    dtemp_dt = data_3d.temp.diff("time") / dt
    dqv_dt = data_3d.qv.diff("time") / dt
    dtemp_dt["time"] = half_time
    dqv_dt["time"] = half_time

    # Add g/c_p * w for Q1
    s_w = G / C_P * data_3d.w
    s_w = s_w.interp(time=half_time)
    ds_dt = (dtemp_dt + s_w).rename("ds_dt")

    # Time average advective tendencies
    dtemp_adv = data_adv.dtemp.interp(time=half_time)
    dqv_adv = data_adv.dqv.interp(time=half_time)

    # Compute Q terms
    q1 = (ds_dt - dtemp_adv).rename("Q1")
    q2 = (dqv_dt - dqv_adv).rename("Q2")

    return xr.merge([q1, q2])


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_3d")
    parser.add_argument("data_adv")
    parser.add_argument("output_zarr")

    args = parser.parse_args()

    data_3d = xr.open_zarr(args.data_3d)
    data_adv = xr.open_zarr(args.data_adv)

    data_out = compute_Q_terms(data_3d, data_adv)

    data_out.to_zarr(args.output_zarr)


if __name__ == "__main__":
    main()
