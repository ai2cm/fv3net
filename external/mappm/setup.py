#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from setuptools import find_packages
from glob import glob
from numpy.distutils.core import Extension, setup

setup(
    author="The Allen Institute for Artificial Intelligence",
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
    description="subroutines from fv_mapz.F90 of the FV3GFS Fortran model",
    install_requires=[],
    license="BSD license",
    include_package_data=True,
    keywords="mappm",
    name="mappm",
    packages=find_packages(),
    version="0.1.0",
    zip_safe=False,
    ext_modules=[Extension(name="mappm.mappm", sources=glob("mappm/*.f90"))],
)
