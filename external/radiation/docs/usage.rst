Usage
=====

Configuration
-------------

The radiation wrapper class ``radiation.Radiation`` is the intended way to run the python radiation scheme. 

It is configured using the configuration class ``radiation.RadiationConfig``. The configuration class can be instantiated in a standalone fashion, or it can be created
from an FV3GFS namelist dictionary using the ``.from_namelist`` method, which ensures that the python radiation scheme will run with the same namelist as in the Fortran radiation scheme of a model run. 

Instantiation
-------------

Once a configuration class object is created, the radiation wrapper can be instantiated from it, along with a few other arguments (physics tracer indices, MPI comm object, model timestep, and model initial time).

.. code-block::

    rad_config = RadiationConfig.from_namelist(fv3gfs_namelist)
    radiation = radiation.Radiation(rad_config, tracer_inds=tracer_inds, comm=comm, timestep=900., init_time=datetime.datetime(2016, 8, 1, 0, 0, 0))


A ``.validate()`` method on the radiation wrapper will confirm that the configurations are valid and implemented in the python port (not all are).

Calling
-------

The radiation wrapper then can be readied for use using the ``.init_driver()`` method, which will download necessary data from a remote host. Finally, the radiation wrapper called as follows:

 .. code-block::

        radiation_outputs = radiation(time, state)

where ``time`` is a datetime-like current time, and ``state`` is a dictionary that must contain the data arrays listed in the wrapper's ``.input_variables`` attribute. The radiation outputs in the resulting dict will be those in the ``.output_variables`` attribute.
