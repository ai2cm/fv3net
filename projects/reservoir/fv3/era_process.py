import numpy as np
import vcm
import vcm.catalog
from vcm.catalog import catalog
import xarray as xr
import netCDF4 as nc
import glob
import cftime
import argparse
import logging
import os

FREGRID_EXAMPLE_SOURCE_DATA = (
    "gs://vcm-ml-raw/2020-11-12-gridspec-orography-and-mosaic-data/C48/*.nc"
)

var_id_mapping = {"sst": "sst", "temp2m": "t2m", "u_wind": "u10", "v_wind": "v10"}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path", help="path to directory /downloaded which contains raw data"
    )
    parser.add_argument("--out_path", help="path to save regridded data")
    parser.add_argument("--variables", nargs="+", help="list of variables to process")
    return parser.parse_args()


# raw nc input files are expected to be in path+var+'/downloaded/
def main(args):
    path = args.in_path
    variables = args.variables
    setup_fregrid()
    for var in variables:
        check_time_steps_complete(path, var)
        weekdays = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        for i, day in enumerate(weekdays):
            merge_into_file(path, var, i, day)
            logger.info(f"{day} is merged")
            regrid_to_cubed_sphere(path, var, day)
            logger.info(f"{day} is regridded")
            interpolated = interpolate_nans(var, day)
            logger.info(f"{day} is interpolated")
            masked = mask(interpolated)
            logger.info(f"{day} is masked")
            masked.to_netcdf(args.out_path + var + "_" + day + ".nc")
            logger.info(f"{day} is done")


def download_source_data():
    os.system("mkdir -p fregrid-example/source-grid-data")
    os.system(
        "gsutil -m cp "
        + FREGRID_EXAMPLE_SOURCE_DATA
        + " fregrid-example/source-grid-data/"
    )


def setup_fregrid():
    download_source_data()
    # create 1x1 grid lat lon grid
    os.system(
        "sudo docker run \
            -v $(pwd)/fregrid-example:/work \
            us.gcr.io/vcm-ml/post_process_run:latest \
            make_hgrid \
            --grid_type regular_lonlat_grid \
            --grid_name /work/era5_lonlat_grid \
            --nxbnds 2 \
            --nybnds 2 \
            --xbnds -180,180 \
            --ybnds -90,90 \
            --nlon 720\
            --nlat 362"
    )

    # create mosaic
    os.system(
        "sudo docker run \
        -v $(pwd)/fregrid-example:/work \
        us.gcr.io/vcm-ml/post_process_run:latest \
        make_solo_mosaic \
        --num_tiles 1\
        --dir /work \
        --tile_file era5_lonlat_grid.nc \
        --mosaic_name /work/era5_lonlat_grid_mosaic"
    )


def check_time_steps_complete(path, var):
    # check if data is complete (all time steps are there)
    time_step_list = []
    for f in glob.glob(path + var + "/downloaded/*.nc"):
        dataset = xr.open_dataset(f)
        time_step_list.append(dataset["time"].data)
    dataset.close()
    arrs = list(np.concatenate(time_step_list))
    arrs.sort()
    for i, date in enumerate(arrs[1:]):
        if not (arrs[i] + np.timedelta64(1, "D") == date):
            raise ValueError(
                "Time series has missing days. Missing day after: {arrs[i], i}"
            )
    logger.info("all time steps checked.")


def merge_into_file(path, var, day_of_week_index, day):
    # create one file per weekday
    for f in glob.glob(path + var + "/downloaded/*.nc"):
        dataset = xr.open_dataset(f)
        data_at_day = dataset.isel(time=slice(day_of_week_index, None)).isel(
            time=slice(None, None, 7)
        )
        dataset.close()

    sorted_data = data_at_day.sortby("time")
    sorted_data.to_netcdf(path + var + "/merged/" + var + "_" + day + ".nc")
    current_working_directory = os.getcwd()
    save_nc_int32_time(
        path + var + "/merged/" + var + "_" + day + ".nc",
        current_working_directory
        + "/fregrid-example/"
        + var
        + "_"
        + day
        + "_i32_time.nc",
    )


def save_nc_int32_time(infile, outfile):
    in_nc = nc.Dataset(infile, "r")

    # load original time variable
    in_time = in_nc.variables["time"]
    in_time_values = in_time[:]
    as_dt = cftime.num2date(in_time_values, in_time.units, calendar=in_time.calendar)
    as_julian = cftime.date2num(as_dt, in_time.units, calendar="julian")
    in_nc.close()

    # save new file without time coordinate
    in_ds = xr.open_dataset(infile)
    in_ds.drop("time").to_netcdf(outfile)

    # add adjusted time dimension to the new file

    out_nc = nc.Dataset(outfile, "a")
    out_time = out_nc.createVariable("time", np.int32, ("time",))
    out_time[:] = as_julian
    for attr in in_time.ncattrs():
        out_time.setncattr(attr, in_time.getncattr(attr))
    out_nc.close()


def regrid_to_cubed_sphere(path, var, day):
    var_id = var_id_mapping[var]

    os.system(
        "sudo docker run \
        -v $(pwd)/fregrid-example:/work \
        us.gcr.io/vcm-ml/post_process_run:latest \
        fregrid \
        --input_mosaic /work/era5_lonlat_grid_mosaic.nc \
        --output_mosaic /work/source-grid-data/grid_spec.nc \
        --input_file /work/"
        + var
        + "_"
        + day
        + "_i32_time.nc \
        --output_file /work/"
        + var
        + "_"
        + day
        + "_cubed.nc \
        --scalar_field "
        + var_id
    )


def interpolate_nans(var, day):
    dt_lis = [
        xr.open_dataset(f, decode_times=False)
        for f in glob.glob("fregrid-example/" + var + "_" + day + "_cubed.*nc")
    ]
    grid = vcm.catalog.catalog["grid/c48"].to_dask()
    new_lis = []
    for d in dt_lis:
        d = d.rename({"lon": "x", "lat": "y"})
        d["y"] = grid["y"]
        d["x"] = grid["x"]
        new_lis.append(d)
    ds = xr.concat(new_lis, dim="tile")
    var_id = var_id_mapping[var]
    cubed = ds[var_id]

    # interpolate nans, because of mitmatch of era5 and c48 land-sea mask mismatch
    cubed = cubed.interpolate_na(dim="x")
    cubed = cubed.interpolate_na(dim="y")
    return cubed


def mask(coarsened):
    land_sea_mask_c48 = catalog["landseamask/c48"].read()
    masked = coarsened.where(land_sea_mask_c48["land_sea_mask"] != 1)
    return masked


if __name__ == "__main__":
    args = add_arguments()
    main(args)
