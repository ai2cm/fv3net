Command-line interface
======================

Train typical machine learning model
------------------------------------

A command-line interface is available using ``python -m fv3fit.train``. For more
detail on the format of the configuration file, see _`configuration`.

.. argparse::
   :module: fv3fit.train
   :func: get_parser
   :prog: python -m fv3fit.train


Training Emulators
------------------

Emulators can be trained with the command line interface ``python -m
fv3fit.train_emulator``. These models load directories of pre-stacked netCDF
files.

.. argparse::
   :module: fv3fit.train_emulator
   :func: get_parser
   :prog: python -m fv3fit.train_emulator

.. warning:: This API is currently experimental and may be merged with
   ``fv3fit.train`` in the future.