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
   :maxdepth: 1
   :caption: Active Workflows:

   coarsen_c384_diagnostics_link
   nudging_link
   end_to_end_link
   prognostic_c48_run_link
   one_step_jobs_link
   diagnostics_to_zarr_link
   fregrid_cube_netcdfs_link
   single_fv3gfs_run_link
   prognostic_run_diags_link
   coarsen_3072_surface_diagnostics_link
   coarsen_restarts_link
   sklearn_regression_link

   
.. toctree::
   :maxdepth: 1
   :caption: Defunct Workflows:

   scale-snakemake_link
   extract_tars_link