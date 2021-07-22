import xarray as xr

PATH = "./state_output.zarr"
NC_PATH = "run_nc_files/state_{date_str}_{tile}.nc"

DT_sec = 900

rn_attrs = {
    "standard_name": "lwe_thickness_of_explicit_precipitation_amount",
    "long_name": "explicit precipitation amount on physics timestep",
    "units": "m",
    "dimensions": ["horizontal_dimension"],
    "type": "real",
    "kind": "kind_phys",
    "intent": "out",
    "optional": "F",
}

def main():
    dat = xr.open_zarr(PATH)

    for i, time in enumerate(dat.time):
        for j, tile in enumerate(dat.tile):
            current = dat.isel(time=i, tile=j)
            fmt_str = "%Y%m%d.%H%M%S"
            outfile = NC_PATH.format(
                date_str=time.values.item().strftime(fmt_str), 
                tile=tile.values.item()
            )
            current.to_netcdf(outfile)

if __name__ == "__main__":
    main()

