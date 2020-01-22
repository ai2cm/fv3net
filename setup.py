from setuptools import find_packages, setup

dependencies = [
    "apache-beam",
    "cloudpickle",
    "dask",
    "gcsfs",
    "fsspec",
    "google-cloud-storage",
    "intake",
    "numba",
    "scikit-image",
    "netCDF4",
    "xarray>=0.14.0",
    "partd",
    "pyyaml>=5.0",
    "xgcm",
    "zarr",
]

setup(
    name="fv3net",
    packages=find_packages(),
    install_requires=dependencies,
    include_package_data=True,
    version="0.1.0",
    description="Improving the GFDL FV3 model physics with machine learning",
    author="Vulcan Inc.",
    license="MIT",
)
