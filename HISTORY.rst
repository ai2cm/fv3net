=======
History
=======


latest
------
* Added physics on end_to_end workflow configuration yaml. Only does baseline run for now.
* Added integration tests (tests/end_to_end_integration) that through CircleCI after image builds
* Made simple step output directory names the default in the orchestrator
* Add `run_with_learned_nudging` workflow
* Update fv3config submodule to v0.3.1
* Add `get_config()` function to fv3net.runtime
* Change API of `diagnostic_to_zarr` workflow so that is saves output zarrs in the given run directory
* Add `nudge_to_obs` module to `kube_jobs`, which helps with the configuration of FV3GFS model runs that are nudged towards GFS analysis


0.1.1 (2020-03-25)
------------------
* Updates to make end-to-end workflow work with fv3atm (fv3gfs-python:v0.3.1)
* Added bump2version for automated versioning of `fv3net` resources
* Add CircleCI build/push capabilities for `fv3net` images


0.1.0 (2020-03-20)
------------------
* First release of fv3net
