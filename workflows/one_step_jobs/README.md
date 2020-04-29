## One Step Jobs

Workflow to perform many short FV3GFS runs initialized from sequential timesteps.

Specific model configurations can be specified through the `one-step-yaml` argument.

Included configurations are:
- `all-physics-off.yml` (model will not do any physics or
microphysics)
- `deep-and-mp-off.yml` (model will not do microphysics or deep convection).

Both of these configurations use a one-minute timestep with no dynamics substepping and
have a total duration of 15 minutes.

This workflow can be submitted with the [orchestrate_submit_jobs.py] script.
This script is self-documenting and its help can be seen by running:

    python orchestrate_submit_jobs.py -h


# Minimal example

Here is a minimal exmaple for how to run this script on a limited set of sample images.

```sh
workdir=$(pwd)
src=gs://vcm-ml-data/orchestration-testing/test-andrep/coarsen_restarts_source-resolution_384_target-resolution_48/
output=gs://vcm-ml-data/testing-noah/one-step
VERSION=<image version>
image=us.gcr.io/vcm-ml/prognostic_run:$VERSION
yaml=$PWD/deep-conv-off.yml

gsutil -m rm -r $output > /dev/null
 (
    cd ../../
    python $workdir/orchestrate_submit_jobs.py \
        $src $yaml $image $output -o  \
	--config-version v0.3
 )

```


# Kubernetes VM access troubleshooting

To process many (> around 40) runs at once, it is recommended to submit this workflow
from a VM authorized with a service account. Users have had issues with API request errors
when submitting from a machine authorized with a non-service Google account.

To submit to the kubernetes cluster on a VM, the kubectl configuration needs to point at a 
proxy cluster access point and the VM needs to have a firewall rule to allow for communication 
with the proxy IP. See the long-lived-infrastructure [cluster access README](https://github.com/VulcanClimateModeling/long-lived-infrastructure#vm-access-setup) 
for details on this process, specifically, the make kubeconfig_init command. Most VMs should 
already have a firewall rule set up, but a good way to test whether everythings working is 
by running the following command:

```
kubectl get pods --all-namespaces
```

If the command hangs, then it's likely the firewall rule is not set up properly or the configuration is not set up correctly.
Use the following command to view your current configuration. It should point towards the external IP address of the k8s-ml-cluster-dev-proxy-lsch VM.

```
kubectl config view
```

# Out of Memory errors

The one step jobs can be fail with OOMKilled errors if too many dask workers
are used. These errors can typically be avoided by using the single-threaded
dask scheduler. You can enable for this debugging purposes by adding the
following lines to the top of [runfile.py](./runfile.py):

    import dask
    dask.config.set(scheduler='single-threaded')
