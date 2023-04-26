# ufs_utils docker image

This directory contains files used in building a docker image containing a
compiled version of the UFS_UTILS codebase.  The method of using `spack` to aid
in building the dependencies in the docker image was derived from work in
[ufs-community/UFS_UTILS#749](https://github.com/ufs-community/UFS_UTILS/pull/749),
and the `restart_files_to_nggps_initial_condition.sh` and
`gfs_sigio_analysis_to_nggps_initial_condition` scripts were heavily based on
[the example `chgres_cube` script contained in the UFS_UTILS
repository](https://github.com/ufs-community/UFS_UTILS/blob/develop/ush/chgres_cube.sh).
Accordingly we include a copy of the license used by UFS_UTILS in this
subdirectory.  We also thank Kai-Yuan Cheng and Jake Huff for sharing how they
use the `chgres_tool` to generate initial conditions from various sources at
GFDL.

## Building the image

The image is not hooked into our continuous integration workflow since we do not
expect the need to update it frequently.  If there are any updates, the image
therefore needs to be rebuilt manually.  This can be done using the existing
make rules within the fv3net Makefile:

```
$ make build_image_ufs_utils
$ make push_image_ufs_utils
```
