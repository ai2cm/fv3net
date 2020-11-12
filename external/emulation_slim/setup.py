#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    "numpy>=1.11",
    "fsspec>=0.6.2",
    "pyyaml>=5.1.2",
    "tensorflow>=2.2.0",
]

setup_requirements = []

test_requirements = []

setup(
    author="Vulcan Technologies LLC",
    author_email="andrep@vulcan.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="emulation_slim for the prognostic emulators",
    install_requires=requirements,
    extras_require={},
    license="BSD license",
    include_package_data=True,
    name="emulation_slim",
    packages=find_packages(include=["emulation_slim", "emulation_slim.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    version="0.1.0",
    zip_safe=False,
)
