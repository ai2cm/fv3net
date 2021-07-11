.. emulation_transforms:

Emulation transforms
====================

Transforms is a module for composable functions to form data processing pipelines.  These are useful, for instance, to perform necessary processing on batched data before use in ML training loops.

The functions take a single argument and return the processed output.  Therefore it currently makes heavy use of ``toolz.functoolz.curry`` to provide partial functions that conform to this single-in/out paradigm.


Transforms
----------

.. automodule:: fv3fit.emulation.data.transforms
    :members:
