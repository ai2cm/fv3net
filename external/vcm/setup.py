#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

install_requirements = [
    "Click>=7.0",
    "f90nml>=1.1.0",
    "appdirs>=1.4.0",
    "requests",
    "h5py>=2.10",
    "dask",
    "xarray",
    "toolz",
    "scipy",
    "scikit-image",
    "metpy",
    "pooch>=1.1.1",
    "numba",
    "intake",
    "gcsfs",
    "zarr",
    "xgcm",
    "cftime",
    "pytest",
    "pytest-regtest",
]


setup(
    author="Vulcan Technologies, LLC",
    author_email="noahb@vulcan.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="vcm contains general purposes tools for analyzing FV3 data",
    install_requires=install_requirements,
    license="BSD license",
    include_package_data=True,
    keywords="vcm",
    name="vcm",
    packages=find_packages(),
    version="0.1.0",
    zip_safe=False,
)
