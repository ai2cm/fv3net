from setuptools import find_namespace_packages, setup

dependencies = [
    "dask==2.13.0",
    "f90nml==1.1.2",
    "fsspec==0.7.1",
    "gcsfs==0.6.1",
    "intake==0.5.4",
    "scikit-image==0.16.2",
    "MetPy==0.12.0",
    "pooch",
    "toolz==0.10.0",
    "xarray==0.15.1",
    "xgcm==0.3.0",
    "zarr==2.4.0",
    "numba",
    "cftime==1.1.1.2",
    "vcm",
]

setup(
    name="fv3net-dataflow",
    packages=find_namespace_packages(include=["fv3net.*"]),
    install_requires=dependencies,
    version="0.2.3",
    description="Improving the GFDL FV3 model physics with machine learning",
    author="Vulcan Inc.",
    license="MIT",
)
