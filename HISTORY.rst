=======
History
=======


latest
------
* Added physics on end_to_end workflow configuration yaml. Only does baseline run for now.
* Added integration tests (tests/end_to_end_integration) that through CircleCI after image builds
* Fixed integration tests to use same version tags of the `fv3net` and `prognostic_run` images
* Added makefile targets to submit integration tests to cluster from local machine and to get docker image names


0.1.1 (2020-03-25)
------------------
* Updates to make end-to-end workflow work with fv3atm (fv3gfs-python:v0.3.1)
* Added bump2version for automated versioning of `fv3net` resources 
* Add CircleCI build/push capabilities for `fv3net` images


0.1.0 (2020-03-20)
------------------
* First release of fv3net
