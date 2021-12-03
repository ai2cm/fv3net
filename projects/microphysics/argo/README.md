# Prognostic Argo Workflows

The argo workflow `prog-run-and-eval.yaml` performs a prognostic
run followed by optional evaluation scripts.

For examples of usage or to run a quick experiment, check out the `Makefile`

## Workflow options

* `tag` - run tag grouping the prognostic run and evaluation. If not unique, this will likely fail if run on the same day due to runfv3 GCS conflict.  Otherwise, the artifacts tied to this tag are updated on wandb
* `config` - base64 encoded configuration used for the prognostic run.  expects read in file contents, e.g., `-p config=$(base64 --wrap 0 fv3config.yaml)`
* `tf_model` - GCS URL of tensorflow model to load for prognostic run
* `image_tag` - the tag of the docker image to use

## WandB Secret

This is the resource manifest to create the API key used by `wandb` in the microphysics workflow.

```
apiVersion: v1
stringData:
  WANDB_API_KEY: <COPY API KEY HERE>
kind: Secret
metadata:
  name: wandb-andrep
  namespace: default
type: Opaque
```

Copy to a file (e.g., `wandb-secret.yaml`) and apply using `kubectl apply -f <filename>`
