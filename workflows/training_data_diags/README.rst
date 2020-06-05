===================
training_data_diags
===================

A workflow step for processing training data into a diagnostics dataset

**Usage**

An example entrypoint is provided for running the module:

./workflows/training_data_diags/submit_job.sh

To see the pipeline module's usage run ``python compute_diags.py -h`` in this directory; example::

    usage: compute_diags.py [-h] datasets_config_yml output_path

    positional arguments:
      datasets_config_yml  Config file with dataset paths, mapping functions, and batch specifications.
      output_path          Local or remote path where diagnostic dataset will be written.