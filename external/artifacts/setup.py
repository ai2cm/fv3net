#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_namespace_packages

install_requirements = [
    "gcsfs",
    "fsspec",
]

setup(
    author="Allen Insitute of Artificial Intelligence",
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
    description="Artifacts CLI",
    entry_points={"console_scripts": ["artifacts=fv3net.artifacts.cli:main"]},
    install_requires=install_requirements,
    license="BSD license",
    include_package_data=True,
    name="fv3net-artifacts",
    packages=find_namespace_packages(include=["fv3net.artifacts"]),
)
