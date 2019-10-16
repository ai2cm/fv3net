
from setuptools import setup

setup(name='coarsen-dataflow',
      version='0.0.1',
      packages=['coarseflow'],
      install_requires=[
          'apache-beam[gcp]==2.16.0',
          'google-cloud-storage==1.20.0',
      ],
      )