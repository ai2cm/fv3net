.. _workflows:

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
    
    readme_links/coarsen_3072_surface_diagnostics_readme
    readme_links/coarsen_c384_diagnostics_readme
    readme_links/coarsen_restarts_readme
    readme_links/coarsen_sfc_data_readme
    readme_links/dataflow_readme
    readme_links/diagnostics_to_zarr_readme
    readme_links/fine_res_budget_readme
    readme_links/offline_ml_diags_readme
    readme_links/post_process_run_readme
    readme_links/prognostic_c48_run_readme
    readme_links/prognostic_run_diags_readme
    readme_links/restarts_to_zarr_readme
    readme_links/training_data_diags_readme
