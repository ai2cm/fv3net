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
    "u10": "u10_at_next_time_step",
    "v10": "v10_at_next_time_step",
}

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path", help="path to directory /var/downloaded which contains raw data"
    )
    parser.add_argument("--out_path", help="path to save regridded data")
    parser.add_argument(
        "--train_val_test", help="whether you are creating train, val or test data"
    )
    parser.add_argument(
        "--start_date", help="start date as a string in the format of YYYY-MM-DD"
    )
    parser.add_argument(
        "--end_date", help="end date as a string in the format of YYYY-MM-DD"
    )
    parser.add_argument("--variables", nargs="+", help="list of variables to process")
    # dates to split val, test
    return parser.parse_args()


# raw nc input files are expected to be in path+var+'/downloaded/
def main(args):
    path = args.in_path
    variables = args.variables
    setup_fregrid()
    for var in variables:
        check_time_steps_complete(path, var)
        merge_into_weekly_file(path, var, args)
        logger.info(f"{var} is merged into weekly file")
        regrid_to_cubed_sphere(path, var)
        logger.info(f"{var} is regridded")
        interpolated = interpolate_nans(var)
        interpolated.to_netcdf("interpolated.nc")
        logger.info(f"{var} is interpolated")
        masked = mask(interpolated)
        masked.to_netcdf("temp.nc")
        logger.info(f"{var} is masked")
        if var in ["temp2m", "u_wind", "v_wind"]:
            shifted = shift_variable(masked)
        else:
            shifted = masked
        shifted.to_netcdf(
            args.out_path + var + "_" + args.train_val_test + "_regridded.nc"
        )
        logger.info(f"{var} is done")
    if "sst" in variables:
        calculate_sst_tendency(args)
        variables.append("sst_tendency")
    merged = merge_variables(variables, args)
    for var in variables:
        var_id = var_id_mapping[var]
        merged = merged.rename_vars({var_id: shift_rename_mapping[var_id]})
    del merged.time.attrs["units"]
    del merged.time.attrs["calendar"]
    save_data_as_tiles(merged, args)


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
                f"Time series has missing days. Missing day after: {arrs[i], i}"
            )
    logger.info("all time steps checked.")


def merge_into_weekly_file(path, var, args):
    merged_data = xr.concat(
        [xr.open_dataset(f) for f in glob.glob(path + var + "/downloaded/*.nc")],
        dim="time",
    )
    merged_data = merged_data.sortby("time")
    # get time chunk according to inout args todo
    sorted_data = merged_data.sel(time=slice(args.start_date, args.end_date))
    # merge into weekly mean
    weekly_merged = calculate_weekly_mean(sorted_data)
    current_working_directory = os.getcwd()
    weekly_merged.to_netcdf(path + var + "/merged/" + var + ".nc")
    save_nc_int32_time(
        path + var + "/merged/" + var + ".nc",
        current_working_directory + "/fregrid-example/" + var + "_i32_time.nc",
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


def regrid_to_cubed_sphere(path, var):
    var_id = var_id_mapping[var]
    command = (
        "sudo docker run \
        -v $(pwd)/fregrid-example:/work \
        us.gcr.io/vcm-ml/post_process_run:latest \
        fregrid \
        --input_mosaic /work/era5_lonlat_grid_mosaic.nc \
        --output_mosaic /work/source-grid-data/grid_spec.nc \
        --input_file /work/"
        + var
        + "_i32_time.nc \
        --output_file /work/"
        + var
        + "_cubed.nc \
        --scalar_field "
        + var_id
    )
    os.system(command)


def interpolate_nans(var):
    file_list = glob.glob("fregrid-example/" + var + "_cubed.*nc")
    file_list.reverse()
    dt_lis = [xr.open_dataset(f, decode_times=False) for f in file_list]
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


def shift_variable(data):
    # rename variable
    return data.shift(time=-1)


def calculate_sst_tendency(args):
    seven_days_in_seconds = 604800
    sst_dataset = xr.open_dataset(
        args.out_path + "sst_" + args.train_val_test + "_regridded.nc",
        decode_times=False,
    )
    sst_resid_data_set = sst_dataset.copy()
    sst_resid_data_set = (
        sst_resid_data_set.shift(time=-1) - sst_resid_data_set
    ) / seven_days_in_seconds
    # rename dataset
    sst_resid_data_set = sst_resid_data_set.rename_vars({"sst": "sst_tendency"})
    # save residual dataset
    sst_resid_data_set.to_netcdf(
        args.out_path + "sst_tendency_" + args.train_val_test + "_regridded.nc"
    )


def merge_variables(variables, args):
    data_list = []
    for var in variables:
        # load data
        data = xr.open_dataset(
            args.out_path + var + "_" + args.train_val_test + "_regridded.nc",
            decode_times=False,
        )
        data_list.append(data)
    return xr.merge(data_list)


def save_data_as_tiles(data, args):
    # save each tile individually
    for i in range(len(data.tile)):
        data.isel(tile=i).to_netcdf(
            args.out_path + args.train_val_test + "_tile_" + str(i) + "_regridded.nc"
        )


if __name__ == "__main__":
    args = add_arguments()
    main(args)
