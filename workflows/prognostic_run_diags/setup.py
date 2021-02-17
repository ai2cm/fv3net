#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_namespace_packages

install_requirements = [
    "xarray",
    "holoviews",
    "fsspec",
    "gcsfs",
    "bokeh",
    "h5netcdf",
    "cftime",
    "intake",
    "numpy",
    "google-cloud-storage",
    "toolz",
    "cartopy",
    "intake-xarray",
    # including these vulcan packages managed with poetry causes this error
    # pkg_resources.DistributionNotFound: The 'fv3viz' distribution was not
    # found and is required by fv3net-diagnostics-prog-run
    # "fv3viz",
    "vcm",
    "report",
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
    description="Prognostic run diagnostics",
    scripts=[
        "scripts/stitch_movie_stills.sh",
        "scripts/memoized_compute_diagnostics.sh",
    ],
    entry_points={
        "console_scripts": [
            "prognostic-run-diags=fv3net.diagnostics.prognostic_run.cli:main"
        ]
    },
    install_requires=install_requirements,
    license="BSD license",
    include_package_data=True,
    name="fv3net-diagnostics-prog-run",
    packages=find_namespace_packages(include=["fv3net.diagnostics.prognostic_run"]),
    version="0.1.0",
)
