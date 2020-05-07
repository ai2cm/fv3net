==============
one_step_diags
==============

A poetry micropackage for creating diagnostic report on the one-step jobs

**Usage**

Entrypoints are provided for launching jobs locally or on Dataflow:

./workflows/one_step_diags/submit_diags_direct.sh

./workflows/one_step_diags/submit_diags_dataflow.sh

The module's usage is:::

    usage: python -m one_step_diags [-h] [--report_directory REPORT_DIRECTORY] [--diags_config DIAGS_CONFIG] [--data_fold DATA_FOLD] [--start_ind START_IND] [--n_sample_inits N_SAMPLE_INITS] [--coarsened_diags_zarr_name COARSENED_DIAGS_ZARR_NAME] one_step_data hi_res_diags timesteps_file netcdf_output

    positional arguments:
      one_step_data         One-step zarr path, not including zarr name.
      hi_res_diags          C384 diagnostics zarr path, not including zarr name.
      timesteps_file        File containing paired timesteps for test set. See documentation in one-steps scripts for more information.
      netcdf_output         Output location for diagnostics netcdf file.

    optional arguments:
      -h, --help            show this help message and exit
      --report_directory REPORT_DIRECTORY
                            (Public) bucket path for report and image upload. If omitted, report iswritten to netcdf_output.
      --diags_config DIAGS_CONFIG
                            File containing one-step diagnostics configuration mapping to guide plot creation. Plots are specified using configurationn in .config.py but additional
                            plots can be added by creating entries in the diags_config yaml.
      --data_fold DATA_FOLD
                            Whether to use 'train', 'test', or both (None) sets of data in diagnostics.
      --start_ind START_IND
                            First timestep index to use in zarr. Earlier spin-up timesteps will be skipped. Defaults to 0.
      --n_sample_inits N_SAMPLE_INITS
                            Number of initalizations to use in computing one-step diagnostics.
      --coarsened_diags_zarr_name COARSENED_DIAGS_ZARR_NAME
                            (Public) bucket path for report and image upload. If omitted, report iswritten to netcdf_output.
