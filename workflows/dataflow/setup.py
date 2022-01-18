from setuptools import find_namespace_packages, setup

dependencies = [
    "dask==2.13.0",
    "f90nml==1.1.2",
    "fsspec",
    "gcsfs==2021.6.0",
    "intake==0.5.4",
    "MetPy==0.12.0",
    "toolz==0.10.0",
    "xarray==0.19.0",
    "xgcm==0.3.0",
    "zarr==2.7.0",
    "numba",
    "cftime==1.2.1",
    "vcm",
    "xpartition==0.2.0",
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
