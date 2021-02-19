.. Prognostic Run documentation master file, created by
   sphinx-quickstart on Thu Jan 21 01:54:32 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Prognostic Run's documentation!
==========================================

A machine-learning capable python wrapper of the FV3 dynamical core. It also
supports nudging workflows either to observations or fine resolution
datasets. This the primary tool used to pre-process datasets and evaluate
machine learning models online.


Configuring and executing prognostic runs currently require using a few command
line utilities:

.. graphviz::

   digraph foo {
      "minimal-fv3config.yml" -> "full fv3config.yml" [label="prepare_config.py"];
      "full fv3config.yml" -> "Segmented run" [label="runfv3 create"];
      "Segmented run" -> "Segmented run" [label="runfv3 append"];
   }

Please see the documentation below:

.. toctree::
   :maxdepth: 2
   :caption: Usage:

   prepare-config
   local-execution
   development

.. toctree::
   :maxdepth: 2
   :caption: Reference:

   configuration
   outputdata



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
