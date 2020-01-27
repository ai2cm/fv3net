Prognostic model run workflow
=============================

This workflow makes makes a coupled C48 FV3 run with a scikit-learn model.
It is mostly organized through the Makefile.

Authenticating with google cloud
--------------------------------

This workflow needs to access google cloud storage within docker. To do this,
it expects that the environmental variable GOOGLE_APPLICATION_CREDENTIALS
will point to a valid service account key on the host system. For example,

    export GOOGLE_APPLICATION_CREDENTIALS=<path/to/key.json>

Quickstart
----------

To build the docker image to used, run

	make build

To start a bash shell in this image, run

	make dev

The prognostic run can be started with 

	make sklearn_run

Saving the fv3gfs-python state as a pickle
------------------------------------------

For prototyping purposes it is useful to be able to save the state as returned
by `fv3gfs.get_state` so it can be loaded outside the docker image. This can be
done by running

	make save_state

This will create a file `state.pkl`, which can be read in python using 
	
```python
import state_io

with open("state.pkl", "rb") as f:
    data = state_io.load(f)
```

The script `run_sklearn.py` reads in this file and prints some outputs.
    make test_run_sklearn
