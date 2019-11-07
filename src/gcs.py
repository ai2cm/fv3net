import logging
import subprocess
import xarray as xr
from dask import delayed
import gcsfs


def authenticate(key):
    logging.debug("authenticating with key at {key}")
    ret = subprocess.call(['gcloud', 'auth', 'activate-service-account', '--key-file', key])
    if ret == 0:
        logging.warning("Authentication failed. could lead to "
                        "errors if no other authentication has been configured")
    else:
        logging.debug("authentication succeeded.")


@delayed
def upload_to_gcs(src, dest, save_op):
    logging.info("uploading %s to %s" % (src, dest))
    subprocess.check_call(['gsutil', '-q', 'cp',  src, dest])
    logging.info("uploading %s done" % dest)


def exists(url):
    proc = subprocess.call(['gsutil', 'ls', url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc == 0


def list_matches(pattern):
    """Call `gsutil ls <pattern>`"""
    files = subprocess.check_output(
        ['gsutil', 'ls', pattern]
    )
    return [arg.decode('UTF-8') for arg in files.split()]


def copy(src, dest):
    logging.debug(f"copying {src} to {dest}")
    command = ['gsutil', '-m', 'cp', '-r', src, dest]
    subprocess.check_call(command)
    logging.debug(f"copying {src} to {dest} done")


def strip_trailing_slash(src: str) -> str:
    return src.rstrip("/")


def copy_directory_contents(src: str, dest):
    return copy(strip_trailing_slash(src) + "/*", dest)


def copy_many(urls, dest):
    command = ['gsutil', '-m', 'cp', '-r'] + urls + [dest]
    subprocess.check_call(command)
    
    
def open_remote_zarr(zarr_path: str, project: str = 'vcm-ml') -> xr.Dataset:
    """Open a zarr dataset on GCS
    
    Args:
        zarr_path: a path to a zarr dataset, beginning with the bucket name
        project (optional): the GCP project name
        
    Returns:
        An xarray dataset of the zarr data 
    
    """
    fs = gcsfs.GCSFileSystem(project, token=None)
    gcsmap = fs.get_mapper(zarr_path)
    return xr.open_zarr(store=gcsmap)

def write_remote_zarr(ds: xr.Dataset, gcs_path: str) -> None:
    """Writes an xarray dataset to a zarr on GCS
    
    Args:
        ds: xarray dataset
        gcs_path: the GCS path to the zarr to be written, beginning with tbe bucket name
    
    """
    fs = gcsfs.GCSFileSystem(project='vcm-ml')
    gcsmap = fs.get_mapper(gcs_path)
    ds.to_zarr(store=gcsmap, mode='w')