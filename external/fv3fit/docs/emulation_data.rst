.. emulation_data_:

Emulation Data Tools
====================

``fv3fit.emulation.data`` contains a set of composable transforms useful for building training input pipelines, an input transform configuration, and convenience functions for creating tensorflow datasets from batched data.

Example
-------

Here we show use of our standard input transformation configuration to go from a directory of netCDF files to grouped per-variable input and target tuples for keras training.

.. code-block:: python

    from fv3fit.emulation.data import get_nc_files, batched_to_tf_dataset, TransformConfig

    transform = TransformConfig(
        from_netcdf_path=True,
        input_variables=["air_temperature", "specific_humdity"],
        output_variables["tendency_of_air_temperature_due_to_fv3_physics"],
        vertical_subselections=dict(specific_humdity=slice(10,None)),
    )

    files = get_nc_files("/path/to/netcdf/directory")

    tf_ds = batched_to_tf_dataset(files, transform)
    

.. toctree::
    :maxdepth: 1
    :caption: Contents

    emulation_transforms
    emulation_data_api