
from setuptools import setup, find_packages

dependencies = [
    "apache-beam==2.16.0",
    "cloudpickle==1.2.2",
    "dask==2.6.0",
    "fsspec==0.5.2",
    "google-cloud-storage==1.20.0",
    # "intake==0.5.3",
    # "intake-xarray==0.3.1",
    "netCDF4==1.4.2",
    # "toolz==0.10.0",
    # "xarray==0.13.0",
    # "zarr==2.3.2",
    "partd"
]
setup(name='coarsen-dataflow',
      version='0.0.1',
      packages=find_packages(),
      install_requires=dependencies
      )