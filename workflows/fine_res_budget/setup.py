# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['budget']

package_data = \
{'': ['*']}

install_requires = []

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
