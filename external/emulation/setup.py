#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    "cftime",
    "f90nml",
    "fsspec",
    "gcsfs",
    "mpi4py",
    "numpy",
    "pyyaml",
    "tensorflow",
    "xarray",
    "zarr",
    "scipy",
    "numba",
    # ai2 deps
    "fv3fit",
]

setup_requirements = []

test_requirements = []

setup(
    author="The Allen Institute for Artificial Intelligence",
    author_email="andrep@allenai.org",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="emulation hooks for call_py_fort from fv3gfs",
    install_requires=requirements,
    extras_require={},
    license="BSD license",
    include_package_data=True,
    name="emulation",
    packages=find_packages(include=["emulation", "emulation.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    version="0.1.0",
    zip_safe=False,
)
