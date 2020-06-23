# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["offline_ml_diags"]

package_data = {"": ["*"]}

install_requires = []

setup_kwargs = {
    "name": "offline_ml_diags",
    "version": "0.1.0",
    "description": "",
    "long_description": None,
    "author": "Anna Kwa",
    "author_email": "annak@vulcan.com",
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.7,<4.0",
}


setup(**setup_kwargs)
