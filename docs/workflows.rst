Workflows
=========

Training and testing machine learning parameterizations is a very complicated 
computation, requiring transforming terabytes of high resolution model output, and
running thousands of initial value experiments with the coarse resolution FV3 model.
That is just the pre-processing before a single machine learning model is trained!

The workflows in this project are described by the following graph.

.. graphviz::

   digraph foo {
      "SHiELD Simulations at GFDL" -> "coarsen restarts"
      "SHiELD Simulations at GFDL" -> "coarsen diagnostics";
      "coarsen restarts" -> "one step runs";
      "one step runs" -> "create training data";
      "coarsen diagnostics" -> "create training data";
      "create training data" -> "train sklearn model";
      "train sklearn model" -> "test sklearn model";
      "train sklearn model" -> "prognostic run";
      "one step runs" -> "prognostic run";
      "prognostic run" -> "prognostic run diagnostics";
   }


.. toctree::
    :caption: List of Workflows:
    :maxdepth: 1
    :glob:
    
    readme_links/*
