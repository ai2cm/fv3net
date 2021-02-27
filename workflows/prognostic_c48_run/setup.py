#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = ["fv3fit", "vcm", "jsonschema>=3.2", "dacite >= 1.6", "fv3config>=0.6.0"]

test_requirements = ["pytest"]

setup(
    author="Vulcan Technologies LLC",
    author_email="noahb@vulcan.com",
    python_requires=">=3.6",
    package_data={"": ["*.json"]},
    description="The prognostic run application code. Not a library.",
    install_requires=requirements,
    name="prognostic_run",
    packages=find_packages(),
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={"docs": ["sphinx >=3.2", "sphinx-rtd-theme>=0.5.0"]},
)
