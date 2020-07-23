fv3viz
------
This package contains visualization tools used by the Vulcan Climate Modeling ML team.
Plotting functions found here have been refactored (without change) from their previous
locations in `vcm.visualize` and `gallery`. Sample usage:

.. code-block:: python

    import fv3viz as viz
    
    fig = viz.plot_cube(viz.mappable_var(ds), ...)

