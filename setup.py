from setuptools import setup, find_packages

dependencies = [
    "apache-beam==2.16.0",
    "cloudpickle==1.2.2",
    "dask==2.6.0",
    "fsspec==0.5.2",
    "google-cloud-storage==1.20.0",
    "intake==0.5.3",
    "scikit-image",
    "netCDF4==1.4.2",
    "xarray==0.13.0",
    "partd",
    "pyyaml==3.13",
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
