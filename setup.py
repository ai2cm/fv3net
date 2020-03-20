from setuptools import find_packages, setup

dependencies = [
    "apache-beam",
    "backoff",
    "cloudpickle",
    "dask",
    "gcsfs",
    "fsspec",
    "google-cloud-storage",
    "h5netcdf",
    "intake",
    "metpy",
    "numba",
    "scikit-image",
    "netCDF4",
    "xarray>=0.14.0",
    "partd",
    "pyyaml>=5.0",
    "xgcm",
    "zarr",
    "kubernetes",
]

setup(
    name="fv3net",
    packages=find_packages(),
    install_requires=dependencies,
    version="0.1.0",
    description="Improving the GFDL FV3 model physics with machine learning",
    author="Vulcan Inc.",
    license="MIT",
)
