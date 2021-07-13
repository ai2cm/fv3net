.. _emulation_data:

Emulation Data Tools
====================

``fv3fit.emulation.data`` contains a set of composable transforms useful for building training input pipelines, an input transform configuration, and convenience functions for creating tensorflow datasets from a sequence of data.

Example
-------

Here we show use of our standard input transformation configuration to go from a directory of netCDF files to grouped per-variable input and target tuples for keras training.

.. code-block:: python

    from fv3fit.emulation.data import nc_dir_to_tf_dataset, TransformConfig

    transform = TransformConfig(
        input_variables=["air_temperature", "specific_humdity"],
        output_variables["tendency_of_air_temperature_due_to_fv3_physics"],
        vertical_subselections=dict(specific_humdity=slice(10,None)),
    )

    tf_ds = nc_dir_to_tf_dataset("/path/to/netcdf/directory", transform)

.. note::
    The standard input transformations (:py:class:`fv3fit.emulation.data.TransformConfig`) expect input data source files or batches to have the shapes [sample x feature] or [sample].  This may not be the always be the case for custom or future standardized pipelines.

.. toctree::
    :maxdepth: 1
    :caption: Contents

    emulation_transforms
    emulation_data_api