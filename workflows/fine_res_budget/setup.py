# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['budget']

package_data = \
{'': ['*']}

install_requires = \
['apache_beam[gcp]>=2.20.0,<3.0.0',
 'fsspec>=0.7.3,<0.8.0',
 'gcsfs>=0.6.2,<0.7.0',
 'mappm',
 'vcm',
 'xarray>=0.15.1,<0.16.0',
 'zarr>=2.4.0,<3.0.0']

setup_kwargs = {
    'name': 'budget',
    'version': '0.1.0',
    'description': '',
    'long_description': None,
    'author': 'Noah D. Brenowitz',
    'author_email': 'nbren12@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7,<4.0',
}


setup(**setup_kwargs)
