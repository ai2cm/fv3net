# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    description="A python port of the GFS radiation scheme",
    name="radiation",
    author="The Allen Institute for Artificial Intelligence",
    author_email="noahb@allenai.org",
    license="BSD license",
    version="0.1.0",
    python_requires=">=3.8.0",
    packages=find_packages(),
    install_requires=["numba>=0.54.1", "xarray", "numpy", "netcdf4"],
)
