loaders
=======


The `loaders` package contains modules that are used to load data for use in
training and offline evaluation workflows. Since the size of the datasets used
is usually larger than will fit in memory, data is processed in batches rather
than in its entirety.

### Mappers
The mapper objects in this package divide datasets along the time dimension, 
with each timestep corresponding to a key. The API section describes public
functions that return mapper objects.

### Batches
The training and offline diagnostics workflows operate on a sequence of batches.
Each item in the sequence is a single `xarray` dataset which fits into memory.
The API section describes public functions that return sequences of batches.
