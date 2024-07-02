#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_namespace_packages

install_requirements = [
    "scipy>=1.5.0",
    "xarray",
    "holoviews",
    "fsspec",
    "gcsfs",
    "bokeh",
    "h5netcdf",
    "cftime",
    "intake",
    "numpy",
    "flox",
    "google-cloud-storage",
    "toolz",
    "cartopy",
    "intake-xarray",
    "fv3viz",
    "vcm",
    "wandb>=0.12.1",
    "report",
    "streamlit",
    "plotly",
    "fv3fit",
    "loaders",
]

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
    ],
    description="Diagnostics for offline evaluation and prognostic runs",
    entry_points={
        "console_scripts": [
            "prognostic_run_diags=fv3net.diagnostics.prognostic_run.cli:main",
        ]
    },
    install_requires=install_requirements,
    license="BSD license",
    include_package_data=True,
    name="fv3net-diagnostics",
    packages=find_namespace_packages(
        include=[
            "fv3net.diagnostics._shared",
            "fv3net.diagnostics.prognostic_run",
            "fv3net.diagnostics.offline",
            "fv3net.diagnostics.reservoir",
        ]
    ),
    version="0.1.0",
)
