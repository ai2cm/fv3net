from src import gcs
import xarray as xr

BASE_PATH = 'vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/zarr_new_dims/C48/'
OUTPUT_PATH = 'vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/big.zarr/'

def to_big_zarr(gcs_base_dir: str, gcs_output_zarr: str, n_zarr: int = None) -> None:
    """Concatenates all of the timestep zarrs in a base directory and uploads them to a one big zarr on GCS;
        written with intention of being run on VM with Xarray/Dask to handle out of core concatenation
    
    Args:
        gcs_base_dir: the gcs path to a directory containing multiple timestep zarrs 
            to be concatenated, starting with the bucket name
        gcs_output_zarr: the path and name of the zarr to be written to GCS, starting with the bucket name
        
    Returns:
        None
    """
    
    zarr_list = gcs.list_matches('gs://' + gcs_base_dir)
    print(f"Found {len(zarr_list) - 1} .zarr directories on path.")
    
    if n_zarr:
        n_zarr = min(n_zarr, len(zarr_list))
    else:
        n_zarr = len(zarr_list)

    print("Grabbing the first zarr...")
    print(f"{zarr_list[1]}")
    big_ds = gcs.open_remote_zarr(zarr_list[1])
    
    for i, zarr in enumerate(zarr_list[2:n_zarr]):
        print(f"{i + 2}: {zarr}...")
        big_ds = xr.combine_by_coords([big_ds, gcs.open_remote_zarr(zarr)])
        print("...stitched")
        
    print("Writing to GCS...")
    gcs.write_remote_zarr(big_ds, gcs_output_zarr)
    print("...finished.")
    
if __name__=='__main__':
    
    to_big_zarr(BASE_PATH, OUTPUT_PATH)