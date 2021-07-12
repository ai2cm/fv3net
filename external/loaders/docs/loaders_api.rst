
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

.. automodule:: loaders
   :imported-members:
   :members:
   :undoc-members:
   :show-inheritance:
