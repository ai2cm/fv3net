===================
training_data_diags
===================

A poetry micropackage for processing training data from multiple sources into a diagnostic dataset

**Usage**

An example entrypoint is provided for running the module:

./workflows/training_data_diags/submit_job.sh

To see the pipeline module's usage run ``python -m training_data_diags -h`` in this directory; example::

    usage: __main__.py [-h] datasets_config_yml output_path

    positional arguments:
      datasets_config_yml  Config file with dataset paths, mapping functions, and batch specifications.
      output_path          Local or remote path where diagnostic dataset will be written.
