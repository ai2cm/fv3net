# Transform Restart Files to Zarr

For post-processing the restart file format is pretty unpleasant.
This PR transforms all of these restarts directories into a big zarr.


## Design

Use dask distributed to do this


## Set up minikube to use local registry

Run on
```
eval $(minikube docker-env)  # bash
eval (minikube docker-env)   # fish
```

Also need to set
```yaml
imagePullPolicy: Never
```

in all the relevant yamls.
