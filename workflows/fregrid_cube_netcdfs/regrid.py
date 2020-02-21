import argparse
import fsspec
import os
import xarray as xr
import tarfile
import sh
import logging
from subprocess import check_call


grid = "gs://vcm-ml-data/2019-10-05-coarse-grid-and-orography-data.tar"

parser = argparse.ArgumentParser()
parser.add_argument("src_prefix")
parser.add_argument("outputBucket")
parser.add_argument("resolution")
parser.add_argument("--scalar_fields")
parser.add_argument("--mosaic", default=grid)
parser.add_argument("--args", default="--nlat 180 --nlon 360")

args = parser.parse_args()


fs = fsspec.filesystem("gs")


def download_tile_data(src_prefix):
    tiles = []
    for k in range(6):
        tile_suffix = f".tile{k+1}.nc"
        tile_path = src_prefix + tile_suffix
        output = "data" + tile_suffix
        logging.info(f"Downloading {tile_path}")
        fs.get(tile_path, output)
        tiles.append(output)
    return tiles


def get_mosaic(mosaic, resolution):
    mosaic = (
        f"2019-10-05-coarse-grids-and-orography-data/{args.resolution}/grid_spec.nc"
    )
    if not os.path.exists(mosaic):
        with fsspec.open(mosaic) as f:
            tar = tarfile.open(fileobj=f)
            tar.extractall()
    return mosaic


def get_scalar_fields(data):
    ds = xr.open_dataset(data[0])
    vars = []
    for var in ds:
        if set(ds[var].dims) >= {"grid_yt", "grid_xt"}:
            vars.append(var)
    return ",".join(vars)


files = download_tile_data(args.src_prefix)
mosaic = get_mosaic(args.mosaic, args.resolution)
remap_file = "remap_file.nc"
scalar_fields = args.scalar_fields if args.scalar_fields else get_scalar_fields(files)

check_call(
    [
        "fregrid",
        "--input_mosaic",
        mosaic,
        "--remap_file",
        remap_file,
        "--input_file",
        "data",
        "--output_file",
        "data.nc",
        "--scalar_field",
        scalar_fields,
    ]
    + args.args.split()
)

logging.info(f"Uploading to {args.outputBucket}")
fs.put("data.nc", args.outputBucket)