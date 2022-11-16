import xarray as xr
from datetime import datetime

if __name__ == "__main__":
    paths = {
        "neg4K": "gs://vcm-ml-experiments/spencerc/2021-06-14/n2f-25km-baseline-minus-4k/fv3gfs_run/",  # noqa: E501
        "0K": "gs://vcm-ml-experiments/spencerc/2021-05-24/n2f-25km-baseline-unperturbed/fv3gfs_run/",  # noqa: E501
        "4K": "gs://vcm-ml-experiments/spencerc/2021-05-24/n2f-25km-baseline-plus-4k/fv3gfs_run/",  # noqa: E501
        "8K": "gs://vcm-ml-experiments/spencerc/2021-08-11/n2f-25km-baseline-plus-8k/fv3gfs_run/",  # noqa: E501
    }
    date_string = datetime.now().strftime("%Y-%m-%d")
    for name, path in paths.items():
        ds_atmos = xr.open_zarr(path + "atmos_dt_atmos.zarr")
        ds_sfc = xr.open_zarr(path + "sfc_dt_atmos.zarr")
        ds = xr.merge([ds_atmos, ds_sfc])
        out_path = f"gs://vcm-ml-experiments/mcgibbon/{date_string}/n2f-25km-{name}-merged.zarr"  # noqa: E501
        print(f"writing zarr to {out_path}")
        ds.to_zarr(out_path, consolidated=True)
