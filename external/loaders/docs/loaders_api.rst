
loaders API
===========

Mappers
-------
We use the following mappers to load data used by diagnostic and ML training routines.
These are python Mappings whose keys correspond to timesteps of the form ``YYYYMMDD.HHMMSS``,
and whose values are the data at the keyed timestep.

.. autodata:: loaders.mapper_functions

These mappers can be retrieved using the :py:class:`loaders.MapperConfig` configuration class.


Batches
-------
A "batch" in the fv3net sense is a dataset that is used in one loop iteration of either a training loop or
an offline diagnostics calculation loop. These workflows use batches in order to utilizer a larger amount of
training/test data than would otherwise fit in memory.

There are two types of functions which can create a sequence of batches, "batches functions" and "batches from mapper functions". The first type, "batches functions" can be initialized using the :py:class:`loaders.BatchesConfig` class, and can use the following functions:

.. autodata:: loaders.batches_functions

The second type can be initialized using the :py:class:`loaders.BatchesFromMapperConfig` class, and can take the following values for ``batches_function`` alongside any of the ``mapper_function`` values above:

.. autodata:: loaders.batches_from_mapper_functions

The functions themselves take a ``data_path`` argument, followed by the keyword arguments in their API documentation below.

Command-line validation
-----------------------

Configuration files for batches data can be validated using a command-line tool ``validate_batches_config`` provided by this package. If the configuration is valid, it will exit without error. Note this only checks for type validation of the configuration entries, and does not test that the data referenced actually exists and is without errors.

.. code-block:: bash

   $ validate_batches_config --help
   usage: validate_batches_config [-h] config

   positional arguments:
   config      path of BatchesLoader configuration yaml file

   optional arguments:
   -h, --help  show this help message and exit


Command-line downloading
------------------------

This package also provides a command-line tool for downloading batches based on a BatchesLoader yaml configuration file to a local directory of netCDF files.

.. argparse::
   :module: loaders.batches.save
   :func: get_parser
   :prog: python3 -m loaders.batches.save


API Reference
-------------

Loaders provides

.. automodule:: loaders
   :imported-members:
   :members:
   :undoc-members:
   :show-inheritance:
