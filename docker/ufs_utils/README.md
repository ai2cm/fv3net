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
at another resolution.  The `chgres_cube` tool can also be used to generate
initial conditions from GFS analysis data.  We currently do not include a script
for doing this in the docker image, but it would be straightforward to add.

## Building the image

The image is not hooked into our continuous integration workflow since we do not
expect the need to update it frequently.  If there are any updates, the image
therefore needs to be rebuilt manually.  This can be done using the existing
make rules within the fv3net Makefile:

```
$ make build_image_ufs_utils
$ make push_image_ufs_utils
```
