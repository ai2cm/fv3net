#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()

test_requirements = ["pytest"]

setup(
    author="The Allen Institute for Artificial Intelligence",
    author_email="elynnw@allenai.org",
    python_requires=">=3.6",
    description="SCREAM prognostic run application code",
    install_requires=requirements,
    name="scream_run",
    packages=find_packages(include=["scream_run", "scream_run.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    entry_points={
        "console_scripts": [
            "write_scream_run_directory=scream_run.cli:write_scream_run_directory",
            "scream_run=scream_run.cli:scream_run",
            "prepare_scream_config=scream_run.cli:prepare_scream_config",
        ]
    },
)
