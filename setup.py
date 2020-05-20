from setuptools import find_packages, setup

dependencies = [
    "dask==2.13.0",
    "f90nml==1.1.2",
    "fsspec==0.7.1",
    "gcsfs==0.6.1",
    "intake==0.5.4",
    "scikit-image==0.16.2",
    "MetPy==0.12.0",
    "pooch==0.1.1",
    "toolz==0.10.0",
    "xarray==0.15.1",
    "xgcm==0.3.0",
    "zarr==2.4.0",
    "numba==0.48.0",
    "cftime==1.1.1.2",
]


setup(
    name="fv3net",
    packages=find_packages(),
    install_requires=dependencies,
    version="0.2.1",
    description="Improving the GFDL FV3 model physics with machine learning",
    author="Vulcan Inc.",
    license="MIT",
)
