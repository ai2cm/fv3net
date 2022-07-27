.. configuration_:

Configuration
=============

Configuration is loaded from yaml files into the :py:class:`fv3fit.TrainingConfig` class for training options and a :py:class:`loaders.BatchesLoader` class for data. For example configuration files, check out the `vcm-workflow-control examples <https://github.com/ai2cm/vcm-workflow-control/tree/master/examples>`_, in particular this `training-config.yaml <https://github.com/ai2cm/vcm-workflow-control/blob/master/examples/train-evaluate-prognostic-run/training-config.yaml>`_

Any artifacts created during training saved in the `artifacts` directory will be copied to the final output location along with the trained model.
i.e. configuration entries that specify output locations for artifacts should be given as `artifacts/...`.

For more information on loaders configuration, see the `loaders documentation <https://vulcanclimatemodeling.com/docs/loaders/loaders_api.html>`

.. autoclass:: fv3fit.TrainingConfig
   :members:
   :undoc-members:
   :noindex:

   .. automethod:: __init__
