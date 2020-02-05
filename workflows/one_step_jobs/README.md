Workflow to perform many short FV3GFS runs initialized from sequential timesteps.

Specific model configurations can be specified through the `one-step-yaml` argument.

Included configurations are:
- `all-physics-off.yml` (model will not do any physics or
microphysics)
- `deep-and-mp-off.yml` (model will not do microphysics or deep convection).

Both of these configurations use a one-minute timestep with no dynamics substepping and
have a total duration of 15 minutes.

Workflow call signature:
```
$ python submit_jobs.py -h
usage: submit_jobs.py [-h] --one-step-yaml ONE_STEP_YAML --input-url INPUT_URL
                      --output-url OUTPUT_URL [--n-steps N_STEPS]

optional arguments:
  -h, --help            show this help message and exit
  --one-step-yaml ONE_STEP_YAML
                        Path to local run configuration yaml.
  --input-url INPUT_URL
                        Remote url to initial conditions. Initial conditions
                        are assumed to be stored as INPUT_URL/{timestamp}/{tim
                        estamp}.{restart_category}.tile*.nc
  --output-url OUTPUT_URL
                        Remote url where model configuration and output will
                        be saved. Specifically, configuration files will be
                        saved to OUTPUT_URL/one_step_config and model output
                        to OUTPUT_URL/one_step_output
  --n-steps N_STEPS     Number of timesteps to process. By default all
                        timesteps found in INPUT_URL for which successful runs
                        do not exist in OUTPUT_URL will be processed. Useful
                        for testing.
```


### Kubernetes VM access troubleshooting

To process many (> around 40) runs at once, it is recommended to submit this workflow
from a VM authorized with a service account. Users have had issues with API request errors
when submitting from a machine authorized with a non-service Google account.

To submit to the kubernetes cluster on a VM, the kubectl configuration needs to point at a proxy cluster access point and the VM needs to have a firewall rule to allow for communication with the proxy IP. See the long-lived-infrastructure cluster access README for details on this process, specifically, the make kubeconfig_init command. Most VMs should already have a firewall rule set up, but a good way to test whether everythings working is by running the following command:

```
kubectl get pods --all-namespaces
```

If the command hangs, then it's likely the firewall rule is not set up properly or the configuration is not set up correctly.
Use the following command to view your current configuration. It should point towards the external IP address of the k8s-ml-cluster-dev-proxy-lsch VM.

```
kubectl config view
```
