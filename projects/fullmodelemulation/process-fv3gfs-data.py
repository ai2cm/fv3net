import os
import subprocess
import tempfile

import xarray as xr
import fsspec

CONFIG = {
    "input_url": "gs://vcm-ml-raw-flexible-retention/2023-03-17-sample-regridded-C96-FV3GFS-FME-data",
    "output_url": "gs://vcm-ml-scratch/oliwm/test-fv3gfs-processing-v1",
    # "output_url": "gs://vcm-ml-raw-flexible-retention/2023-03-21-sample-regridded-C96-FV3GFS-FME-data-merged-with-some-fluxes",
    "variables": {
        "fourcastnet_vanilla_1_degree": (
            "UGRD10m",
            "VGRD10m",
            "TMP2m",
            "PRESsfc",
            "PRMSL",
            "UGRD1000",
            "VGRD1000",
            "h1000",
            "TMP850",
            "UGRD850",
            "VGRD850",
            "h850",
            "RH850",
            "TMP500",
            "UGRD500",
            "VGRD500",
            "h500",
            "RH500",
            "h50",
            "TCWV",
            "HGTsfc",
        ),
        "full_state_1_degree": (
            "surface_temperature",
            "PRATEsfc",
            "LHTFLsfc",
            "SHTFLsfc",
            "DSWRFsfc",
            "DLWRFsfc",
            "USWRFsfc",
            "ULWRFsfc",
            "DSWRFtoa",
            "USWRFtoa",
            "ULWRFtoa",
        ),
    },
}

def gsutil_cp(source, destination):
    subprocess.check_call(["gsutil", "cp", source, destination])

def glob_file_lists(fs, url, categories):
    files = {}
    for diagnostic_category in categories:
        files_cat = fs.glob(os.path.join(url, diagnostic_category, "*.nc"))
        files[diagnostic_category] = sorted((os.path.basename(f) for f in files_cat))
    return files


def assert_values_equal(dict_):
    # ensure all items in dict_ have the same value
    first_key = list(dict_)[0]
    first_val = dict_[first_key]
    for val in dict_.values():
        assert val == first_val, (
            "There exist netCDF files with different "
            f"names in different categories. Got {val} and {first_val}."
        )


def process_file(basename, input_url, output_url, variables):
    print(f"Processing {basename}...")
    ds_list = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for category in variables:
            gcs_path = os.path.join(input_url, category, basename)
            local_path = os.path.join(tmpdir, category, basename)
            gsutil_cp(gcs_path, local_path)
            print(f"Opening data from {local_path}")
            tmp = xr.open_dataset(local_path)
            tmp = tmp[list(variables[category])]
            ds_list.append(tmp)
        ds = xr.merge(ds_list)
        output_local_path = os.path.join(tmpdir, 'outputfile.nc')
        print("Writing merged data to ", output_local_path)
        ds.to_netcdf(output_local_path)
        gsutil_cp(output_local_path, os.path.join(output_url, basename))


def main(config, limit_to_single_file=False):
    fs, _, _ = fsspec.get_fs_token_paths(config["input_url"])
    files = glob_file_lists(fs, config["input_url"], list(config["variables"]))
    assert_values_equal(files)
    if limit_to_single_file:
        files = {k: [v[0]] for k, v in files.items()}

    for file in files[list(files)[0]]:
        process_file(
            file, config["input_url"], config["output_url"], config["variables"]
        )


if __name__ == "__main__":
    main(CONFIG)
