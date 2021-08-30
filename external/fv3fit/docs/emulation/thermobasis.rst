.. _thermobasis:

Thermodynamics-aware ML
-----------------------


This package contains thermodynamics aware objects that can be used for building
machine learning models. 


Thermodynamic Bases
~~~~~~~~~~~~~~~~~~~

.. module:: fv3fit.emulation.thermobasis.thermo

The core data structure of this subpackage is the :py:class:`ThermoBasis` data
structure.

These objects provide two features, allow switching between different
representations of the thermodynamic state (e.g. relative humidity or specific
humidity). They also include type hints which enable static typechecking and
auto-completion.

Currently, there are two implementations of this interface.

.. autoclass:: SpecificHumidityBasis

.. autoclass:: RelativeHumidityBasis


Models
~~~~~~

.. automodule:: fv3fit.emulation.thermobasis.models
    :members:


Loss functions
~~~~~~~~~~~~~~

.. automodule:: fv3fit.emulation.thermobasis.loss
    :members: