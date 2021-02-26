Grid operations
===============

Horizontal coarse-graining
--------------------------

VCM provides a number of convenience functions for doing various forms of
"horizontal block reduction," useful for coarsening data to lower resolution.

General purpose block reductions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. automethod:: vcm.cubedsphere.block_coarsen

Weighted block averages
~~~~~~~~~~~~~~~~~~~~~~~

These two functions are frequently used for coarse-graining state variables of
the dynamical core, e.g. temperature or the horizontal winds.

   .. automethod:: vcm.cubedsphere.weighted_block_average
   .. automethod:: vcm.cubedsphere.edge_weighted_block_average

Block edge sums
~~~~~~~~~~~~~~~

The primary use-case for this function is for coarse-graining cell edge lengths.

   .. automethod:: vcm.cubedsphere.block_edge_sum

Custom xarray block reductions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Xarray's coarsenening functionality supports many block reduction types, but not
all.  These functions are xarray and dask compatible wrappers around
`scikit-image's block_reduce function <https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.block_reduce>`_, which enable coarsening with custom
reduction functions.

   .. automethod:: vcm.cubedsphere.horizontal_block_reduce
   .. automethod:: vcm.cubedsphere.xarray_block_reduce

Upsampling a reduced field to a fine-resolution grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Occasionally it is useful to take a coarse field and repeat it back out such
that it has the same resolution as the fine grid.
:py:func:`vcm.cubedsphere.block_upsample` is designed for this purpose.

   .. automethod:: vcm.cubedsphere.block_upsample


Interpolation
-------------

General purpose interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. automethod:: vcm.interpolate.interpolate_1d
   .. automethod:: vcm.interpolate.interpolate_unstructured


Interpolating to globally constant pressure levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. automethod:: vcm.interpolate.interpolate_to_pressure_levels


Interpolation using the vertical remapping algorithm of FV3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. automethod:: vcm.cubedsphere.regrid_vertical


Regional calculations
---------------------

.. automodule:: vcm.select
   :members:
   :undoc-members:
   :show-inheritance:
