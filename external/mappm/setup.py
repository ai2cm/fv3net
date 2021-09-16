#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from numpy.distutils.core import Extension, setup

install_requirements = ["numpy"]


setup(
    author="Allen Institue of Artificial Intelligence",
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
    description="Vertical mapping routines from FV3GFS",
    install_requires=["numpy"],
    license="BSD license",
    name="mappm",
    version="0.1.0",
    zip_safe=False,
    ext_modules=[Extension(name="mappm", sources=["mappm.f90", "interpolate_2d.f90"])],
)
