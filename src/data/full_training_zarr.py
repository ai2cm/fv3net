from src import gcs
import xarray as xr
from datetime import datetime
from src.data.calc import apparent_source, compute_tendency
from urllib.error import HTTPError
from apache_beam.utils import retry


BASE_PATH = "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/zarr_new_dims/C48/"
OUTPUT_PATH = f"gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/full_training_zarr/{datetime.now().isoformat()}.zarr"
N_ZARR = None
TENDENCY_VARS = ["sphum", "T", "u", "v", "W"]
OUTPUT_CHUNKS = {'tile' : 6, 'initialization_time' : 1, 'forecast_time' : 1}


@retry.with_exponential_backoff()
def process(counter, zarr):
    print(f"{counter}: stitching {zarr}...", flush=True)
    return gcs.open_remote_zarr(zarr)


def add_tendencies_and_residuals(stitched_ds):
    """Adds tendencies and residual tendencies to output dataset of restarted model runs
    
    Args:
        stitched_ds: dataset for which tendencies and residuals should be added
        
    Returns:
        stitched_ds: dataset with tendencies and residuals added
    """

    print("Computing tendencies and residuals...", flush=True)

    for var in TENDENCY_VARS:
        stitched_ds[f"d{var}_dt"] = compute_tendency(
            stitched_ds[var], t_dim="initialization_time"
        )
        stitched_ds[f"d{var}_dt_C48"] = compute_tendency(stitched_ds[var], t_dim="forecast_time")

    stitched_ds["Q1"] = apparent_source(stitched_ds.T)
    stitched_ds["Q2"] = apparent_source(stitched_ds.sphum)

    print("...finished computing tendencies and residuals.", flush=True)
    
    return stitched_ds


def redo_chunks(ds, new_chunks):
    """Reset chunking in dataset prior to writing to storage
    
    Args:
        ds: dataset to be rechunked
        chunks: dict defining chunk sizes along dimensions to be rechunked
        
    Returns:
        ds: rechunked ds
    """
    
    for var in ds.data_vars:
        if 'chunks' in ds[var].encoding:
            del ds[var].encoding['chunks']
    
    return ds.chunk(new_chunks)
    


def to_full_training_zarr(
    gcs_base_dir: str, gcs_output_zarr: str, n_zarr: int = None
) -> None:
    """Concatenates all of the timestep zarrs in a base directory and uploads them to a one big zarr on GCS;
        written with intention of being run on VM with Xarray/Dask to handle out of core concatenation
    
    Args:
        gcs_base_dir: the gcs path to a directory containing multiple timestep zarrs 
            to be concatenated
        gcs_output_zarr: the path and name of the zarr to be written to GCS
        
    Returns:
        None
    """

    print(f"Starting process of creating {gcs_output_zarr}")
    
    zarr_list = gcs.list_matches(gcs_base_dir)
    print(f"Found {len(zarr_list) - 1} .zarr directories on path.")

    if n_zarr:
        n_zarr = min(n_zarr, len(zarr_list))
    else:
        n_zarr = len(zarr_list)

    print(f"Starting to stitch {n_zarr - 1} datasets...")

    stitched_ds = xr.combine_by_coords(
        [process(i, zarr) for i, zarr in enumerate(zarr_list[1:n_zarr])]
    )

    stitched_ds = add_tendencies_and_residuals(stitched_ds)
    rechunked_ds = redo_chunks(stitched_ds, OUTPUT_CHUNKS)
    
    print("Writing to GCS...", flush=True)
    gcs.write_remote_zarr(rechunked_ds, gcs_output_zarr)
    print("...finished.")


if __name__ == "__main__":

    to_full_training_zarr(BASE_PATH, OUTPUT_PATH, N_ZARR)
