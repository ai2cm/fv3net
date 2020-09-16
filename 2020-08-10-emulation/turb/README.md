# Generate Training Data

The `argo.yaml` provides a workflow to generate turbulence parameterization training
data using SerialBox and `fv3gfs-fortran`.

Submit the workflow using 
```
argo submit turb/argo.yaml -f turb/turb_params.yaml
```

## Generate necessary docker images

There are two required images in this workflow, the `fv3gfs` image with GCS, SerialBox,
and `fv3gfs-fortran`, and `to-zarr` which contains GCS, SerialBox, and python scripts for
converting serialized data to zarr.

### Generate fv3gfs-fortran image

```
git clone git@github.com:VulcanClimateModeling/fv3gfs-fortran.git --branch ml-emulation

cd fv3gfs-fortran
make build_serialize COMPILED_TAG_NAME=ml-emulation
docker push us.gcr.io/vcm-ml/fv3gfs-compiled:ml-emulation-serialize
```

### Generate physics_standalone image

```
git clone git@github.com:VulcanClimateModeling/physics_standalone.git --branch feature/serialized-to-zarr

cd physics_standalone
make build IMAGE_TAG=ml-emulation
make push IMAGE_TAG=ml-emulation
```


