## Argo Workflow Templates

Argo is a project for orchestrating containers, that we use for long-running
scientific workflows. This folder contains "WorkflowTempates" that can be
installed onto a K8s cluster. Once installed on a cluster, they can be
referenced from other argo workflows, or run directly using the `argo`
command line tool.

### Quickstart

To install these templates run

    kubectl apply -k <this directory>

This can be done from an external location (e.g. vcm-workflow-control). To run an installed workflowtemplate, 
use the `--from` flag. Workflow parameters can be passed via the command line with the `-p` option. For example
```
argo submit --from workflowtemplate/prognostic-run-diags \
    -p runs="$(< rundirs.json)" \
    --name <name>
```

This job can be monitored by running

    argo get <name>

### Pinning the image tags

These workflows currently refer to following images without using any tags:
1. us.gcr.io/vcm-ml/fv3net
1. us.gcr.io/vcm-ml/fv3fit
1. us.gcr.io/vcm-ml/prognostic_run
1. us.gcr.io/vcm-ml/post_process_run

However, you can and should pin these images using kustomize (>=v3). For
example, consuming configurations (e.g. in vcm-workflow-control) could use
the following kustomization.yaml to pin these versions:

```
apiVersion: kustomize.config.k8s.io/v1beta1
resources:
- <path/to/fv3net/workflows/argo>
kind: Kustomization
images:
- name: us.gcr.io/vcm-ml/fv3fit
  newTag: 6e121e84e3a874c001b3b8d1b437813c9859e078
- name: us.gcr.io/vcm-ml/fv3net
  newTag: 6e121e84e3a874c001b3b8d1b437813c9859e078
- name: us.gcr.io/vcm-ml/post_process_run
  newTag: 6e121e84e3a874c001b3b8d1b437813c9859e078
- name: us.gcr.io/vcm-ml/prognostic_run
  newTag: 6e121e84e3a874c001b3b8d1b437813c9859e078
```

It is also possible to do this programmatically, using `kustomize edit set image`.
See the [end-to-end integration tests](/tests/end_to_end_integration/run_test.sh) for an example.

### Running fv3gfs with argo

The `prognostic-run` template is a workflow to do fv3gfs simulations on the
cloud. It can do baseline (no-ML) runs, nudged runs or prognostic runs.
It does post-processing on the fly and the workflow can run the model in
sequential segments.

| Parameter            | Description                                                                   |
|----------------------|-------------------------------------------------------------------------------|
| `initial-condition`  | Timestamp string for time at which to begin the prognostic run                |
| `config`             | String representation of base config YAML file; supplied to prepare_config.py |
| `reference-restarts` | Location of restart data for initializing the prognostic run                  |
| `output`             | Location to save prognostic run output                                        |
| `flags`              | (optional) extra command line flags for prepare_config.py                     |
| `segment-count`      | (optional) Number of prognostic run segments; default "1"                     |
| `cpu`                | (optional) Number of cpus to request; default "6"                             |
| `memory`             | (optional) Amount of memory to request; default 6Gi                           |

#### Command line interfaces used by workflow
This workflow calls:
```
python3 /fv3net/workflows/prognostic_c48_run/prepare_config.py \
        {{inputs.parameters.flags}} \
        {{inputs.parameters.config}} \
        {{inputs.parameters.reference-restarts}} \
        {{inputs.parameters.initial-condition}} \
        > /tmp/fv3config.yaml
```
And then
```
runfv3 create {{inputs.parameters.output}} /tmp/fv3config.yaml /fv3net/workflows/prognostic_c48_run/sklearn_runfile.py
```
Followed by `segment-count` iterations of
```
runfv3 append {{inputs.parameters.output}}
```

#### Volumes used by run-fv3gfs template

The `prognostic-run` template uses an internal workflow template `run-fv3gfs`. Due to 
some limitations of argo, it is necessary that the entrypoint workflow makes a
claim for volumes that are ultimately mounted and used by `run-fv3gfs`. See the
`volumes` section of the `prognostic-run` workflow for an example of the volume claims 
necessary to use `run-fv3gfs`.

### Prognostic run report

The `prognostic-run-diags` workflow template will generate reports for
prognostic runs. See this [example][1].

| Parameter     | Description                                                      |
|---------------|------------------------------------------------------------------|
| `runs`        | A json-encoded list of {"name": ..., "url": ...} items           |
| `make-movies` | (optional) whether to generate movies. Defaults to "false".      |
| `flags`       | (optional) flags to pass to `prognostic_run_diags save` command. |

To specify what verification data use when computing the diagnostics, use the `--verification`
flag. E.g. specifying the argo parameter `flags= --verification nudged_c48_fv3gfs_2016` will use a
year-long nudged-to-obs C48 run as verification. By default, the `40day_may2020` simulation
is used as verification. Datasets in the [vcm catalog](/external/vcm/vcm/catalog.yaml) with
the `simulation` and `category` metadata tags can be used.

The prognostic run report implements some basic caching to speed the generation of multiple
reports that use the same run. The diagnostics and metrics for each run will be saved
to `gs://vcm-ml-archive/prognostic_run_diags/{cache-key}` where `cache-key` is the run url
without the `gs://` part and with forward slashes replaced by dashes. The workflow will only
compute the diagnostics if they don't already exist in the cache. If you wish to force a
recomputation of the diagnostics, simply delete everything under the appropriate cached
subdirectory.

#### Command line interfaces used by workflow
For each `run` in the `runs` JSON parameter, this workflow calls
```
memoized_compute_diagnostics.sh {{run.url}} \
                                gs://vcm-ml-public/argo/{{workflow.name}}/{{run.name}} \
                                {{inputs.parameters.flags}}
```
and then regrids the relevant diagnostics to a lat-lon grid using `cubed-to-latlon.regrid-single-input`.
It then optionally calls
```
prognostic_run_diags movie {{run.url}} /tmp/movie_stills
stitch_movie_stills.sh /tmp/movie_stills  gs://vcm-ml-public/argo/{{workflow.name}}/{{run.name}}
```
Once these steps are completed, a report is generated with
```
prognostic_run_diags report gs://vcm-ml-public/argo/{{workflow.name}} gs://vcm-ml-public/argo/{{workflow.name}}
```

#### Workflow Usage Example

Typically, `runs` will be stored in a json file (e.g. `rundirs.json`).
```
[
  {
    "url": "gs://vcm-ml-scratch/oliwm/2020-04-27-advisory-council-with-more-features/dsmp-off-cs-solar-phis-ts/prognostic_run",
    "name": "dsmp-off-cs-solar-phis-ts"
  },
  {
    "url": "gs://vcm-ml-scratch/oliwm/2020-04-27-advisory-council-with-more-features/physics-on-solar-phis-ts/prognostic_run",
    "name": "physics-on-solar-phis-ts"
  }
]
```

You can create a report from this json file using the following command from a bash shell:
```
argo submit --from workflowtemplate/prognostic-run-diags \
    -p runs="$(< rundirs.json)" \
    --name <name>
```

If successful, the completed report will be available at
`gs://vcm-ml-public/argo/<name>/index.html`, where `<name>` is the name of the created
argo `workflow` resource. This can be accessed from a web browser using this link:

    http://storage.googleapis.com/vcm-ml-public/argo/<name>/index.html 

[1]: http://storage.googleapis.com/vcm-ml-public/experiments-2020-03/prognostic_run_diags/combined.html


### training workflow

This workflow trains machine learning models.

| Parameter              | Description                                          |
|------------------------|------------------------------------------------------|
| `input`                | Location of dataset for training data                |
| `config`               | Model training config yaml                           |
| `times`                | JSON-encoded list of timestamps to use for test data |
| `offline-diags-output` | Where to save offline diagnsostics                   |
| `report-output`        | Where to save report                                 |
| `memory`               | (optional) memory for workflow. Defaults to 6Gi.     |

#### Command line interfaces used by workflow
This workflow calls
```
python -m fv3fit.train \
          {{inputs.parameters.input}} \
          {{inputs.parameters.config}} \
          {{inputs.parameters.output}} \
          --timesteps-file {{inputs.parameters.times}} \
          {{inputs.parameters.flags}}
```

### offline-diags workflow

This workflow computes offline ML diagnostics and generates an associated report.

| Parameter              | Description                                          |
|------------------------|------------------------------------------------------|
| `ml-model`             | URL to machine learning model                        |
| `times`                | JSON-encoded list of timestamps to use for test data |
| `offline-diags-output` | Where to save offline diagnsostics                   |
| `report-output`        | Where to save report                                 |
| `memory`               | (optional) memory for workflow. Defaults to 6Gi.     |

#### Command line interfaces used by workflow
This workflow calls
```
python -m offline_ml_diags.compute_diags \
          {{inputs.parameters.ml-model}} \
          {{inputs.parameters.offline-diags-output}} \
          --timesteps-file {{inputs.parameters.times}} 
          
python -m offline_ml_diags.create_report \
          {{inputs.parameters.offline-diags-output}} \
          {{inputs.parameters.report-output}} \
          --commit-sha "$COMMIT_SHA"
```

### train-diags-prog workflow template

This workflow template runs the `training`, `offline-diags`, `prognostic-run` and
`prognostic-run-diags.diagnostics-step` workflow templates in sequence.

| Parameter               | Description                                                                         |
|-------------------------|-------------------------------------------------------------------------------------|
| `root`                  | Local or remote root directory for the outputs from this workflow                   |
| `train-test-data`       | Location of data to be used in training and testing the model                       |
| `training-config`       | String representation of a training configuration YAML file                         |
| `train-times`           | List strings of timesteps to be used in model training                              |
| `test-times`            | List strings of timesteps to be used in offline model testing                       |
| `public-report-output`  | Location to write HTML report of model's offline diagnostic performance             |
| `initial-condition`     | Timestamp string for time at which to begin the prognostic run                      |
| `prognostic-run-config` | String representation of a prognostic run configuration YAML file                   |
| `reference-restarts`    | Location of restart data for initializing the prognostic run                        |
| `flags`                 | (optional) extra command line flags for prognostic run; passed to prepare_config.py |
| `segment-count`         | (optional) Number of prognostic run segments; default "1"                           |
| `cpu-prog`              | (optional) Number of cpus for prognostic run; default "6"                           |
| `memory-prog`           | (optional) Memory for prognostic run; default 6Gi                                   |
| `memory-training`       | (optional) Memory for model training; default 6Gi                                   |
| `memory-offline-diags`  | (optional) Memory for offline diagnostics; default 6Gi                              |
| `training-flags`        | (optional) extra command line flags for training; passed to fv3fit.train            |

### train-diags-prog-multiple-models workflow template

This is similar to the above `train-diags-prog` workflow, but trains and runs offline diagnostics for >1
ML model, then uses all trained models in the prognostic run. All parameters are the same as for the 
`train-diags-prog` workflow, except this workflow takes a `training-configs` parameter instead of 
`training-config`. `training-configs` is the string representation of a JSON file, which should be formatted as 
`[{name: model_name, config: model_config}, ...]`, and where the model config values are identical in
structure to the single configurations used in `train-diags-prog`.  In practice it is easiest to write this as
a YAML file since our existing training configs are YAMLs that can be pasted in, and then converted to JSON
format using `yq . config.yml` in the submit command.

### Cubed-sphere to lat-lon interpolation workflow

The `cubed-to-latlon` workflow can be used to regrid cubed sphere FV3 data using GFDL's `fregrid` utility.
In this workflow, you specify the input data (the prefix before `.tile?.nc`), the destination
for the regridded outputs, and a comma separated list of variables to regrid from the source file.

| Parameter       | Description                                                              | Example                         |
|-----------------|--------------------------------------------------------------------------|---------------------------------|
| `source_prefix` | Prefix of the source data in GCS (everything but .tile1.nc)              | gs://path/to/sfc_data (no tile) |
| `output-bucket` | URL to output file in GCS                                                | gs://vcm-ml-data/output.nc      |
| `resolution`    | Resolution of input data (defaults to C48)                               | one of 'C48', 'C96', or 'C384'  |
| `fields`        | Comma-separated list of variables to regrid                              | PRATEsfc,LHTFLsfc,SHTFLsfc      |
| `extra_args`    | Extra arguments to pass to fregrid. Typically used for target resolution | --nlat 180 --nlon 360           |
