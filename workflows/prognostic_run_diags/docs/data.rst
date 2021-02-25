.. _data:

Data Requirements
=================

Summary
^^^^^^^

The scripts require that provided prognostic runs have been post-processed
so that their outputs are available in zarr format. Prognostic runs of any
resolution that is a multiple of C48 can be handled by this report. However,
it is possible that the grid for a particular resolution may need to be added
to the catalog to allow coarsening. Verification data must be pre-coarsened
to C48 resolution.


Catalog entries
^^^^^^^^^^^^^^^

Computing diagnostics requires certain entries in an intake catalog. By default,
the ``vcm`` catalog is used. The catalog is assumed to
contain the entries ``grid/c48``, ``grid/c96``, and ``landseamask/c48``.

Custom verification data can be added to the catalog. The entries should include
the following metadata items:

#. ``simulation``, a short unique tag
#. ``category``, which should be either ``dycore`` or ``physics`` depending on which set of diagnostics the catalog entry 
   corresponds to, and
#. ``grid`` which must be ``c48``.

By default, the diagnostics calcluation uses the ``40day_may2020`` simulation as
verification. These catalog entries are shown below as examples::

   40day_c48_gfsphysics_15min_may2020:
    description: 2D physics diagnostics variables from 40-day nudged simulation (May 27 2020 SHiELD), coarsened to C48 resolution and rechunked to 96 15-minute timesteps (1 day) per chunk
    driver: zarr
    metadata:
      grid: c48
      simulation: 40day_may2020
      category: physics
    args:
      storage_options:
        project: 'vcm-ml'
        access: read_only
      urlpath: "gs://vcm-ml-experiments/2020-06-17-triad-round-1/coarsen-c384-diagnostics/coarsen_diagnostics/gfsphysics_15min_coarse.zarr"
      consolidated: True

  40day_c48_atmos_8xdaily_may2020:
    description: 2D dynamical core diagnostics variables from 40-day nudged simulation (May 27 2020 SHiELD), coarsened to C48 resolution.
    driver: zarr
    metadata:
      grid: c48
      simulation: 40day_may2020
      category: dycore
    args:
      storage_options:
        project: 'vcm-ml'
        access: read_only
      urlpath: "gs://vcm-ml-experiments/2020-06-17-triad-round-1/coarsen-c384-diagnostics/coarsen_diagnostics/atmos_8xdaily_C3072_to_C384.zarr"
      consolidated: True
