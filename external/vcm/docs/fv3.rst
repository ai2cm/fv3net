Working with FV3 outputs
========================

FV3GFS restart files
--------------------

I/O
~~~

Restart files can be loaded from disk with the functions

.. autofunction:: vcm.fv3_restarts.open_restarts


Coarse-graining
~~~~~~~~~~~~~~~

VCM Tools provides two high-level methods for coarse-graining a complete set of
restart files produced from running FV3GFS.  The two methods perform the
coarse-graining in different ways.

.. autofunction:: vcm.cubedsphere.coarsen_restarts_on_sigma
.. autofunction:: vcm.cubedsphere.coarsen_restarts_on_pressure

Parsing FV3 logs
----------------

Use the following routines to parse statistical information from the console outputs of an FV3 run:

.. autofunction:: vcm.fv3.logs.loads


Standardizing FV3 outputs
-------------------------

The diagnostics output by FV3GFS have verbose, and sometimes difficult to
understand dimension names.  The functions here provide convenient ways for
renaming dimensions to and from standard names, as well as standardizing time
coordinate information:

.. automodule:: vcm.fv3.metadata
   :members:
   :undoc-members:
   :show-inheritance:
