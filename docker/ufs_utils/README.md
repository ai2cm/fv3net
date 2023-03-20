# UFS_UTILS docker image

This directory contains files used in building a docker image containing a
compiled version of the UFS_UTILS codebase.  The method of using `spack` to aid
in building the dependencies in the docker image was derived from work in
[ufs-community/UFS_UTILS#749](https://github.com/ufs-community/UFS_UTILS/pull/749),
and the `restart_files_to_nggps_initial_condition.sh` script was heavily based
on [the example `chgres_cube` script contained in the UFS_UTILS
repository](https://github.com/ufs-community/UFS_UTILS/blob/develop/ush/chgres_cube.sh).
We include a copy of the license used by UFS_UTILS in this subdirectory.

In terms of functionality the `restart_files_to_nggps_initial.sh` script enables
transforming a set of FV3GFS restart files to an NGGPS-style initial condition
at another resolution.  The `chgres_cube` tool can be used to generate initial
conditions from GFS analysis data; however we currently do not include a script
for doing this in the docker image, but it would be straightforward to add.

## Building the image

The image is not hooked into our continuous integration workflow since we do not
expect the need to update it frequently.  If there are any updates, the image
therefore needs to be rebuilt manually.

If you have modified `base.Dockerfile` then you will need to first update the
base image:
```
$ docker build -f base.Dockerfile -t us.gcr.io/vcm-ml/ufs-utils-base .
```

If not, you can skip straight to building the main image.  After the main image
builds, you can then give it a tag (in place of `$NEW_TAG`) and push it to the
container registry:
```
$ docker build -f Dockerfile -t us.gcr.io/vcm-ml/ufs-utils --build-arg BASE_IMAGE=us.gcr.io/vcm-ml/ufs-utils-base:latest .
$ docker tag us.gcr.io/vcm-ml/ufs-utils:latest us.gcr.io/vcm-ml/ufs-utils:$NEW_TAG
$ docker push us.gcr.io/vcm-ml/ufs-utils:$NEW_TAG
```

We split the image into two parts to ease development since rebuilding what is
in the base image can take up to an hour, and unnecessary rebuilds are often
triggered due to cache misses (see discussion for a similar issue in building
the prognostic run image in
[fv3net#1775](https://github.com/ai2cm/fv3net/issues/1775) and
[fv3net#1831](https://github.com/ai2cm/fv3net/pull/1831)).
