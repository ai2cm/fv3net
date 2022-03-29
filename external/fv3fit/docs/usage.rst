Usage
=====

TransformedModel
----------------

The model :py:class:`fv3fit.train_microphysics.TransformedParameters` used for
microphysics emulation is general purpose and can be used with loaders based training

It has several useful features:

* tensor_transformations: Users can configure and code arbitrary pre-ML and post-ML transformations. The design is easy to extend. This:

  * allows baking data-trained pre-processing logic into the tensorflow graph without special handling
  * allows training with constraints (e.g. qv>0) active.
  * Supports multiple target learning: can use any transformed value in a loss function. This makes it easy to e.g. include Q1, Q2, and QM  in the loss function even if the ML architecture only outputs Q1/Q2. Just add more `loss.loss_variables`. Currently only supports MSE based loss, but that would be easy to extend.

* Different architectures: RNN, dense, etc...also extensible.


Here is an example:
.. literalinclude:: transformed-config.yaml
    :language: yaml