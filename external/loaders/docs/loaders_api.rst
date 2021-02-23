
loaders API
===========


Mappers
-------------------------------
We use the following mappers to load data used by diagnostic and ML training routines.
These are python Mappings whose keys correspond to timesteps of the form `YYYYMMDD.HHMMSS`,
and whose values are the data at the keyed timestep.

.. automodule:: loaders.mappers
   :members: open_nudge_to_obs, open_nudge_to_fine, open_nudge_to_fine_multiple_datasets


Batches
-------------------------------
A "batch" in the fv3net sense is a dataset that is used in one loop iteration of either a training loop or
an offline diagnostics calculation loop. These workflows use batches in order to utilizer a larger amount of 
training/test data than would otherwise fit in memory.

The following functions create sequences of batches.

.. automodule:: loaders.batches
   :members: batches_from_geodata, batches_from_mapper, batches_from_serialized, diagnostic_batches_from_geodata
