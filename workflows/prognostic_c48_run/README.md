Prognostic model run workflow
=============================

This workflow makes makes a coupled C48 FV3 run with a scikit-learn model. This workflow mainly
1. builds a docker image suitable for online prognostic runs with fv3gfs, and
1. includes runfile `sklearn_runfile.py` which can be used with `fv3run`

(Authenticating) with google cloud
--------------------------------

This workflow needs to access google cloud storage within docker. To do this,
it expects that the environmental variable GOOGLE_APPLICATION_CREDENTIALS
will point to a valid service account key on the host system. For example,

    export GOOGLE_APPLICATION_CREDENTIALS=<path/to/key.json>

Quickstart
----------

To build the docker image to used, run

	make build

The git revision of the fv3net installed in the docker image can be specified using the `FV3NET_VERSION` flag. For example,

    FV3NET_VERSION=master make build
    
will install the master branch of fv3net.

To start a bash shell in this image, run

	make dev

An example prognostic run can be started with 

	make sklearn_run
    
Saving the fv3gfs-python state as a pickle
------------------------------------------

For prototyping purposes it is useful to be able to save the state as returned
by `fv3gfs.get_state` so it can be loaded outside the docker image. This can be
done by running

	make state.pkl

This will create a file `state.pkl`, which can be read in python using 
	
```python
import state_io

with open("state.pkl", "rb") as f:
    data = state_io.load(f)
```

The main method of the  `online_modules/sklearn_interface` reads in this file and prints some outputs.

    make test_run_sklearn

Note that the `online_modules` must be in the PYTHONPATH for this make rule to work.

Configure the scikit-learn run
------------------------------------------

The scikit-learn model and location for zarr output can be configured using `fv3config.yml`. To do this, simply add/modifly the `scikit_learn` entry of the yaml file as follows:
```
scikit_learn:
  model: gs://vcm-ml-data/test-annak/ml-pipeline-output/2020-01-17_rf_40d_run.pkl
  zarr_output: diags.zarr
```
