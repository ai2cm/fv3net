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
    "lark>=1.1.1",
    "xarray",
    "toolz",
    "scipy",
    "metpy",
    "joblib",
    "intake",
    "gcsfs",
    "zarr",
    "xgcm",
    "cftime",
    "pytest",
    "google-cloud-storage>=1.18.1",
    "google-api-core",
    "pytest-regtest",
    "h5netcdf>=0.8",
    "intake-xarray>=0.3.1",
    "typing_extensions",
    "dacite",
]
tests_require = ["mappm", "synth"]

setup(
    author="The Allen Institute for Artificial Intelligence",
    author_email="noahb@allenai.org",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="vcm contains general purposes tools for analyzing FV3 data",
    install_requires=install_requirements,
    tests_require=tests_require,
    license="BSD license",
    include_package_data=True,
    keywords="vcm",
    name="vcm",
    packages=find_packages(),
    package_data={"vcm": ["catalog.yaml"]},
    version="0.1.0",
    zip_safe=False,
)
