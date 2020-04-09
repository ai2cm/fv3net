## Run with learned nudging workflow

This workflow (in `workflows/run_with_learned_nudging`) allows an external nudging
tendency be applied to FV3GFS runs.
There are configurations implemented to apply monthly mean nudging tendency of
temperature; temperature and pressure thickness; and finally temperature,
pressure thickness and horizontal winds. See Makefile for examples of how to
submit the jobs.

The `postprocess.sh` script must be called from the root of the fv3net repository.
Furthermore, it must be manually called after the jobs initiated by
`make run_all_remote` finish. Those jobs can be monitored by `kubectl get pods`.
For more detailed info, try `kubectl describe job <jobname>` where the
`<jobname>`s will be printed to the console by `make run_all_remote`.

The docker image for this workflow can be built by calling
`make build_image_learned_nudging_run` in the root of the `fv3net` repo.
