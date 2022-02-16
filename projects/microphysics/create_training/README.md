# Create Zhao-Carr Microphysics Training Dataset

From this folder we can recreate the Zhao-Carr microphysics training
dataset from scratch by running the FV3GFS simulations and gathering
the saved netcdfs.

## FV3GFS Runs

It's set up to perform GFS-initialized C48 runs
starting on the first of each month. The duration of each run is 12
days with output netCDFs (and Zarrs) checkpointed every 5 hours.
It saves the entire dictionary state in call-py-fort, so see the
fv3gfs-fortran for details on the included fields.

To run all simulations from the `create_training` directory:

    make create

You can use the following to tweak the training outputs

* `OUTPUT_FREQUENCY`: output frequency of netcdfs in seconds (default is 18000)
* `TAG_PREFIX`: Tag prefix is used to determine the project output folder

## Gathering netdf files

Each run has its own output folder.  To move all the netcdfs into the training
folders use the following:

    make gather

The netCDFs will be gathered into the base project folder for a given run date
(e.g., 2021-11-18) under `training_netcdfs`.  Set `RUN_DATE` or `TAG_PREFIX`
while using the make operation to adjust the source of the gathered netCDFs.



