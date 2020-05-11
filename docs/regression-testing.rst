Regression Testing
==================

The package ``synth`` provides tools for generating synthetic xarray datasets for testing purposes.
Compared to real data, it is much easier and cheaper to manage code for generating synthetic 
using software version control tools like git.


Usage
-----

This packages allows the user to build up a "schema" describing their data. 
A schema is a reduced description of a zarr/xarray dataset that can
1. be easily serialized to disk and loaded again
1. used to generate a random dataset
1. validated an existing dataset (not implemented yet)

This package defines a set of dataclasses defining a schema.


The function :func:`synth.read_schema_from_zarr` can be used to generate a schema 
from an existing dataset. For example, suppose we have loaded a zarr group like this::

    import sys
    import fsspec
    import zarr
    import synth

    url = "gs://path/to/data.zarr"
    mapper = fsspec.get_mapper(url)
    group = zarr.open_group(mapper)

This data could comprise many gigabytes, so it is unweildy to manage, and use
within a testing framework. To generate a condensed description, you can
generate a reduced "schema" like this::

    def sample(arr):
        return arr[-1, 0, 0]

    schema = synth.read_schema_from_zarr(
        group, sample=sample, coords=("time", "tile", "grid_xt", "grid_yt")
    )

Note that, the ``sample`` should just be a fast way to generate a list of
representative samples from zarr Array. The maximum and minimum over this
sample will be used to define range of values in the schema. Since the input
data in this case has the dimension order ``["time", "tile", "grid_xt",
"grid_yt"]``, the sample function above will simply return the vector along
the "grid_yt" dimension. It is the responsibility of the user to define a
"sample" function that is appropriate for their needs. It is recommended to
avoid loading data from multiple chunks within a sampling function.

Next, ``schema`` can be serialized to json and dumped to disk like
this::

    with open("schema.json", "w") as f:
        synth.dump(schema, f)

The json file can then be checked into version control and loaded inside of a
test script like this::

    with open("schema.json" , "r") as f:
        schema = synth.load(f)
    
Finally, a fake xarray dataset can be created::

    ds = schema.generate()

This data will be constrained to lie within the bounds detected by the
``sample`` function above.


Marking pytest functions as regression tests
--------------------------------------------

Regression tests are identified using `pytest.mark.regression`. These will
only be triggered if the fv3net unit tests succeed.

Examples
--------

The "create training data" workflow currently is tested using this framework.
See test `here <https://github.com/VulcanClimateModeling/fv3net/blob/be447a44725d7fb766bbe35685862246f06f37f9/tests/create_training_data/test_integration.py#L1>`_.


Existing tools
--------------

Python has some `rich tools <https://faker.readthedocs.io/en/master/>`_ for
generating fake data, but nothing specialized to xarray/zarr.


