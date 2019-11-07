from setuptools import find_packages, setup

dependencies = [
    "apache-beam==2.16.0",
    "cloudpickle==1.2.2",
    "dask==2.6.0",
    "fsspec==0.5.2",
    "google-cloud-storage==1.20.0",
    "intake==0.5.3",
    "scikit-image", 
    "intake-xarray==0.3.1",
    "gcsfs",
    "netCDF4==1.4.2",
    "xarray==0.13.0",
    "zarr==2.3.2",
    "partd",
]

setup(
    name="src",
    packages=find_packages(),
    install_requires=dependencies,
    version="0.1.0",
    description="This workflow converts stepped C48 Fv3 runs to zarr files.",
    author="Vulcan Inc.",
    license="MIT",
)
