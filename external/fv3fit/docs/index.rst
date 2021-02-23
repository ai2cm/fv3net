fv3fit
======

fv3fit models should be subclassed from either ``fv3fit.Predictor`` or ``fv3fit.Estimator``. The former defines the interface required by the prognostic run and offline reports to load and predict with a model. The latter defines the methods needed to train and save a model. See the implementations of these classes for more details.

The package provides :py:func:`fv3fit.dump` and :py:func:`fv3fit.load` functions to save and load model classes. New (internal) estimators should be decorated using `fv3fit._shared.io.register`.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   commandline
   configuration
   api

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
