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
import atexit
import shutil

# todo: add land mask

FREGRID_EXAMPLE_SOURCE_DATA = (
    "gs://vcm-ml-raw/2020-11-12-gridspec-orography-and-mosaic-data/C48/*.nc"
)

TIME_SHIFTED_VARIABLES = ["t2m", "u10", "v10"]

var_id_mapping = {
    "sst": "sst",
    "sst_tendency": "sst_tendency",
    "temp2m": "t2m",
    "u_wind": "u10",
    "v_wind": "v10",
}

shift_rename_mapping = {
    "sst": "sst",
    "sst_tendency": "sst_tendency",
    "t2m": "t2m_at_next_timestep",
    "u10": "u10_at_next_timestep",
    "v10": "v10_at_next_timestep",
}

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# example execution: python era_process.py
# --out_path /home/paulah/data/era5/fvtest/val/
# --start_date 2006-01-01 --end_date 2006-03-31
# --variables sst temp2m u_wind v_wind --in_path /home/paulah/data/era5/


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path", help="path to directory /var/downloaded which contains raw data"
    )
    parser.add_argument(
        "--out_path",
        help="path to save regridded data, should include whether train/val/test",
    )
    parser.add_argument(
        "--start_date", help="start date as a string in the format of YYYY-MM-DD"
    )
    parser.add_argument(
        "--end_date", help="end date as a string in the format of YYYY-MM-DD"
    )
    parser.add_argument("--variables", nargs="+", help="list of variables to process")
    parser.add_argument(
        "--fill_value_sst", default=291, type=float, help="value to fill sst nans"
    )
    parser.add_argument(
        "--fill_value_sst_tend",
        default=1.6e-10,
        type=float,
        help="value to fill sst tend nans",
    )
    # dates to split val, test
    return parser.parse_args()


# raw nc input files are expected to be in path+var+'/downloaded/
def main(args):
    tempdirname = "temp"
    os.system("mkdir -p " + tempdirname)
    atexit.register(shutil.rmtree, tempdirname)
    path = args.in_path
    variables = args.variables
    setup_fregrid(tempdirname)
    # merge variables
    for var in variables:
        check_time_steps_complete(path, var)
    full_variables = merge_into_weekly_file(path, variables, args, tempdirname)
    # do time shift, add tendency by doing xarray operations
    # on the full dataset, no need to write it in between steps

    for var in TIME_SHIFTED_VARIABLES:
        full_variables[var] = full_variables[var].shift(time=-1)
    # masking, etc.
    regrid_to_cubed_sphere(full_variables, variables, tempdirname)
    interpolated = interpolate_nans(variables, tempdirname)
    interpolated["sst"] = mask(interpolated["sst"], args)
    if "sst" in variables:
        variables.append("sst_tendency")
        interpolated["sst_tendency"] = calculate_sst_tendency(interpolated.sst)
        interpolated["sst_tendency"].fillna(args.fill_value_sst_tend)
    for var in variables:
        var_id = var_id_mapping[var]
        interpolated = interpolated.rename_vars({var_id: shift_rename_mapping[var_id]})
    save_data_as_tiles(interpolated, args)


def download_source_data(tempdirname):
    os.system("mkdir -p " + tempdirname + "/fregrid-example/source-grid-data")
    os.system(
        "gsutil -m cp "
        + FREGRID_EXAMPLE_SOURCE_DATA
        + " "
        + tempdirname
        + "/fregrid-example/source-grid-data/"
    )


def setup_fregrid(tempdirname):
    download_source_data(tempdirname)
    # create 1x1 grid lat lon grid
    os.system(
        "sudo docker run \
            -v $(pwd)/"
        + tempdirname
        + "/fregrid-example:/work \
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
        -v $(pwd)/"
        + tempdirname
        + "/fregrid-example:/work \
        us.gcr.io/vcm-ml/post_process_run:latest \
        make_solo_mosaic \
        --num_tiles 1\
        --dir /work \
        --tile_file era5_lonlat_grid.nc \
        --mosaic_name /work/era5_lonlat_grid_mosaic"
    )


def check_time_steps_complete(path, var):
    arrs = list(
        np.concatenate(
            [
                xr.open_dataset(f)["time"].data
                for f in glob.glob(path + var + "/downloaded/*.nc")
            ]
        )
    )
    arrs.sort()
    for i, date in enumerate(arrs[1:]):
        if not (arrs[i] + np.timedelta64(1, "D") == date):
            raise ValueError(
                f"Time series has missing days. Missing day after: {arrs[i], i}"
            )
    logger.info("all time steps checked.")


def merge_into_weekly_file(path, variables, args, tempdir):
    for var in variables:
        weekly_merged = xr.concat(
            [
                xr.open_dataset(f)
                for f in glob.glob(os.path.join(path, var + "/downloaded/*.nc"))
            ],
            dim="time",
        )
        weekly_merged = weekly_merged.sortby("time")
        # get time chunk according to inout args todo
        weekly_merged = weekly_merged.sel(time=slice(args.start_date, args.end_date))
        # merge into weekly mean
        weekly_merged = calculate_weekly_mean(weekly_merged)
        # current_working_directory = os.getcwd()
        weekly_merged_path = os.path.join(tempdir, var + "_weekly_mean.nc")
        weekly_merged.to_netcdf(weekly_merged_path)
        nc_int32_path = var + "_i32_time.nc"
        save_nc_int32_time(
            weekly_merged_path, os.path.join(tempdir, nc_int32_path),
        )
        logger.info(f"{var} is merged into weekly file")
        weekly_merged.close()

    return xr.merge(
        [
            xr.open_dataset(os.path.join(tempdir, var + "_i32_time.nc"))
            for var in variables
        ]
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
        if attr == "calendar":
            value = "julian"
        else:
            value = in_time.getncattr(attr)
        out_time.setncattr(attr, value)
    out_nc.close()


def regrid_to_cubed_sphere(full_variables, variables, tempdir):
    full_variables.to_netcdf(tempdir + "/fregrid-example/full_variables.nc")
    full_variables.close()
    var_id_commas_string = " "
    for var in variables:
        var_id_commas_string += var_id_mapping[var] + ","
    var_id_commas_string = var_id_commas_string[:-1]
    command = (
        "sudo docker run \
        -v $(pwd)/"
        + tempdir
        + "/fregrid-example:/work \
        us.gcr.io/vcm-ml/post_process_run:latest \
        fregrid \
        --input_mosaic /work/era5_lonlat_grid_mosaic.nc \
        --output_mosaic /work/source-grid-data/grid_spec.nc \
        --input_file  /work/full_variables.nc \
        --output_file /work/full_cubed.nc \
        --scalar_field "
        + var_id_commas_string
    )
    os.system(command)
    logger.info("is regridded")


def interpolate_nans(variables, tempdir):
    file_list = glob.glob(tempdir + "/fregrid-example/full_cubed.*nc")
    file_list.sort()
    dt_lis = [xr.open_dataset(f, decode_times=False) for f in file_list]
    grid = vcm.catalog.catalog["grid/c48"].to_dask()
    new_lis = []
    for d in dt_lis:
        d = d.rename({"lon": "x", "lat": "y"})
        d["y"] = grid["y"]
        d["x"] = grid["x"]
        new_lis.append(d)
    ds = xr.concat(new_lis, dim="tile")
    for var in variables:
        var_id = var_id_mapping[var]
        cubed = ds[var_id]
        # interpolate nans, because of mitmatch of era5 and c48 land-sea mask mismatch
        cubed = cubed.interpolate_na(dim="x")
        cubed = cubed.interpolate_na(dim="y")
    logger.info("is interpolated")
    return ds


def mask(coarsened, args):
    land_sea_mask_c48 = catalog["landseamask/c48"].read()
    masked = coarsened.where(land_sea_mask_c48["land_sea_mask"] != 1)
    masked.fillna(args.fill_value_sst)
    logger.info("is masked")
    return masked


def calculate_weekly_mean(sorted_data):
    weekly_merged = xr.concat(
        [
            sorted_data.isel(time=slice(i * 7, (i + 1) * 7)).mean("time")
            for i in range(int(sorted_data.time.shape[0] / 7))
        ],
        dim="time",
    )
    n = len(weekly_merged["time"])
    weekly_merged["time"] = sorted_data["time"][::7][:n]
    return weekly_merged


def calculate_sst_tendency(sst_dataset):
    seven_days_in_seconds = 604800
    sst_resid_data_set = sst_dataset.copy()
    sst_resid_data_set = (
        sst_resid_data_set.shift(time=-1) - sst_resid_data_set
    ) / seven_days_in_seconds
    return sst_resid_data_set


def save_data_as_tiles(data, args):
    # save each tile individually in seperate folder

    for i in range(len(data.tile)):
        os.system("mkdir -p " + os.path.join(args.out_path, "tile-" + str(i)))
        data.isel(tile=i).to_netcdf(
            os.path.join(args.out_path, "tile-" + str(i), "regridded.nc")
        )


if __name__ == "__main__":
    args = add_arguments()
    main(args)
