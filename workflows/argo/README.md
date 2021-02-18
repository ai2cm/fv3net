## Argo Workflow Templates

Argo is a project for orchestrating containers, that we use for long-running
scientific workflows. This folder contains "WorkflowTempates" that can be
installed onto a K8s cluster. Once installed on a cluster, they can be
referenced from other argo workflows, or run directly using the `argo`
command line tool.


### Quickstart

To install these templates run

    kubectl apply -k <this directory>

This can be done from an external location (e.g. vcm-workflow-control)

Running a job from a workflowtemplate is similar to running a standard argo
workflow, but uses the `--from` flag instead. For example,

    argo submit --from workflowtemplate/<templatename> ...


Workflow parameters can be passed via the command line, e.g.
```
argo submit --from workflowtemplate/prognostic-run-diags \
    -p runs="$(< rundirs.json)" \
    --name <name>
```

You can also use submit by supplying a file containing the parameters with the `--parameter-file` or `-f` flag, e.g.
```
argo submit --from workflowtemplate/train-diags-prog --parameter-file config.json
```

This command will make output like this:

    Name:                prognostic-run-diags-sps8h
    Namespace:           default
    ServiceAccount:      default
    Status:              Pending
    Created:             Thu Apr 30 12:01:17 -0700 (now)


This job can be monitored by running

    argo watch <Name>

Moreover, the templates within this workflows can be used by other workflows.

### Pinning the image tags

These workflows currently refer to following images without using any tags:
1. us.gcr.io/vcm-ml/fv3net
1. us.gcr.io/vcm-ml/fv3fit
1. us.gcr.io/vcm-ml/prognostic_run
1. us.gcr.io/vcm-ml/post_process_run

However, you can and should pin this images using kustomize (>=v3). For
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
See the [end-to-end intergration tests](/tests/end_to_end_integration) for an example.

### Running fv3gfs with argo

The `run-fv3gfs` template is a general purpose workflow to do fv3gfs simulations on the
cloud. It does post-processing on the fly and the workflow can run the model in
sequential segments to increase reliability and reduce the memory requirement for
the post-processing step. See the prognostic run workflow at
`workflows/argo/prognostic-run.yaml` for an example usage of the `run-fv3gfs`
template.

| Parameter            | Description                                                                                           |
|----------------------|-------------------------------------------------------------------------------------------------------|
| fv3config            | String representation of an fv3config object                                                          |
| runfile              | String representation of an fv3gfs runfile                                                            |
| output-url           | GCS url for outputs                                                                                   |
| cpu                  | (optional) Requested cpu for run-model step                                                           |
| memory               | (optional) Requested memory for run-model step                                                        |
| segment-count        | (optional) Number of segments to run                                                                  |
| working-volume-name  | (optional) Name of volume for temporary work. Volume claim must be made prior to run-fv3gfs workflow. |

Defaults for optional parameters can be found in the workflow.

#### Running multiple segments

The workflow will submit `segment-count` model segments in sequence. The post-processed diagnostic 
outputs from each segment will automatically be appended to the previous segment's at
`output-url`. All other outputs (restart files, logging, etc.) will be saved to
`output-url/artifacts/{timestamp}` where `timestamp` corresponds to the start time of
each segment. The duration of each segment is defined by the `fv3config` object passed
to the workflow.

#### Post-processing and chunking

The post-processing can convert netCDF diagnostic outputs of the form `name.tile?.nc`
to zarr with user-specified chunks and rechunk zarrs output by fv3gfs-wrapper. To
specify that a set of netCDF outputs should be converted to zarr, their chunking must be
defined in the `fortran_diagnostics` section of the config provided to the `prognostic-run`
workflow. The chunking for zarr outputs from the python runfile are defined with the `diagnostics`
section. For example:
```yaml
initial_conditions: gs://vcm-ml-data/initial-conditions-url
namelist:
  coupler_nml:
    days: 5
fortran_diagnostics:
  - name: atmos_8xdaily.zarr
    chunks:
      time: 8
  - name: nudging_tendencies.zarr
    chunks:
      time: 1
  - name: sfc_dt_atmos.zarr
    chunks:
      time: 96
diagnostics:
  - name: diags.zarr
    chunks:
      time: 96
    variables:
      - net_heating
      - net_moistening
      - column_integrated_dQu
      - column_integrated_dQv
```

Some diagnostics have default chunking which is inserted by the `prepare_config.py`
script if no diagnostics are specified.

WARNING: if `segment-count` is greater than 1, the chunk size in time must evenly
divide the length of the time dimension for each diagnostic output.

As a rule of thumb, make sure the size of netCDF outputs is no larger than about
1 GB per file. The size of output files can be controlled by output frequency, 
number/dimensionality of variables in each diagnostic category, and segment length.

#### Volumes used by run-fv3gfs template

Due to some limitations of argo, it is necessary that the entrypoint workflow makes a
claim for volumes that are ultimately mounted and used by `run-fv3gfs`. The name of these
volumes can be passed to the `run-fv3gfs` template. See the end-to-end test workflow at
`tests/end_to_end_integration/argo.yaml` for an example of the volume claims (including 
`gcp-secret-key`) necessary to use `run-fv3gfs` .


### train-diags-prog workflow template

This workflow template runs the model training, offline diagnostics, prognostic run,
and online diagnostics steps, using the following workflow templates: `training`,
`offline-diags`, and `prognostic-run`. Model training can be run with either `sklearn`
or `keras` training routines using the `train-routine` input parameter and passing
an appropriate `training-config` string.

| Parameter             | Description                                                                |
|-----------------------|----------------------------------------------------------------------------|
| root                  | Local or remote root directory for the outputs from this workflow          |
| train-routine         | Training routine to use: e.g., "sklearn" (default) or "keras"              |
| train-test-data       | Location of data to be used in training and testing the model              |
| training-config       | String representation of a training configuration YAML file                |
| train-times           | List strings of timesteps to be used in model training                     |
| test-times            | List strings of timesteps to be used in offline model testing              |
| public-report-output  | Location to write HTML report of model's offline diagnostic performance    |
| initial-condition     | String of initial time at which to begin the prognostic run                |
| prognostic-run-config | String representation of a prognostic run configuration YAML file          |
| reference-restarts    | Location of restart data for initializing the prognostic run               |
| flags                 | (optional) extra command line flags for prepare_config.py                  |
| segment-count         | (optional) Number of prognostic run segments; default 1                    |
| cpu-prog              | (optional) Number of cpus for prognostic run nodes; default 6              |
| memory-prog           | (optional) Memory for prognostic run nodes; default 6Gi                    |
| work-volume-name      | (optional) Working volume name, prognostic run; default 'work-volume'      |

### train-diags-prog-multiple-models workflow template

This is similar to the above `train-diags-prog` workflow, but trains and runs offline diagnostics for >1
model, then uses all trained models in the prognostic run. This allows different outputs to be trained
with separate sets of hyperparameters. All parameters are the same as for the `train-diags-prog` workflow,
except this workflow takes a `training-configs` parameter instead of `training-config`. `training-configs`
is the string representation of a JSON file, which should be formatted as 
`[{name: model_name, config: model_config}, ...]`, and where the model config values are identical in
structure to the single configurations used in `train-diags-prog`.  In practice it is easiest to write this as
a YAML file since our existing training configs are YAMLs that can be pasted in, and then converted to JSON
format using `yq . config.yml` in the submit command. 

 Models and offline diagnostics are saved in "{{inputs.parameters.root}}/trained_models/{{item.name}}" and 
 "{{inputs.parameters.root}}/offline_diags/{{item.name}}".


| Parameter             | Description                                                                |
|-----------------------|----------------------------------------------------------------------------|
| root                  | Local or remote root directory for the outputs from this workflow          |
| train-routine         | Training routine to use: e.g., "sklearn" (default) or "keras"              |
| train-test-data       | Location of data to be used in training and testing the model              |
| training-configs      | String representation of list of training configurations and their names   |
| train-times           | List strings of timesteps to be used in model training                     |
| test-times            | List strings of timesteps to be used in offline model testing              |
| public-report-output  | Location to write HTML report of model's offline diagnostic performance    |
| initial-condition     | String of initial time at which to begin the prognostic run                |
| prognostic-run-config | String representation of a prognostic run configuration YAML file          |
| reference-restarts    | Location of restart data for initializing the prognostic run               |
| flags                 | (optional) extra command line flags for prepare_config.py                  |
| segment-count         | (optional) Number of prognostic run segments; default 1                    |
| cpu-prog              | (optional) Number of cpus for prognostic run nodes; default 6              |
| memory-prog           | (optional) Memory for prognostic run nodes; default 6Gi                    |
| work-volume-name      | (optional) Working volume name, prognostic run; default 'work-volume'      |


### Prognostic run report

The `prognostic-run-diags` workflow template will generate reports for
prognostic runs. See this [example][1].

| Parameter    | Description                                                  |
|--------------|--------------------------------------------------------------|
| runs         | A json-encoded list of {"name": ..., "url": ...} items       |
| make-movies  | (optional) whether to generate movies. Defaults to false     |
| flags        | (optional) flags to pass to save_prognostic_diags.py script. |

The outputs will be stored at the directory
`gs://vcm-ml-public/argo/<workflow name>`, where `<workflow name>` is NOT the
name of the workflow template, but of the created `workflow` resource.

To specify what verification data use when computing the diagnostics, use the `--verification`
flag. E.g. specifying the argo parameter `flags="--verification nudged_c48_fv3gfs_2016` will use a
year-long nudged-to-obs C48 run as verification. By default, the `40day_may2020` simulation
is used as verification (see fv3net catalog).

The prognostic run report implements some basic caching to speed the generation of multiple
reports that use the same run. The diagnostics and metrics for each run will be saved
to `gs://vcm-ml-archive/prognostic_run_diags/{cache-key}` where `cache-key` is the run url
without the `gs://` part and with forward slashes replaced by dashes. The workflow will only
compute the diagnostics if they don't already exist in the cache. If you wish to force a
recomputation of the diagnostics, simply delete everything under the appropriate cached
subdirectory.

#### Command line Usage Example

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
`gs://vcm-ml-public/argo/<name>/index.html`. This can be accessed from a web browser using this link:

    http://storage.googleapis.com/vcm-ml-public/argo/<name>/index.html 

If you wish to generate movies of column-integrated heating and moistening along with the report, 
add the parameter `-p make-movies="true"`. By default, the movies will not be created.

[1]: http://storage.googleapis.com/vcm-ml-public/experiments-2020-03/prognostic_run_diags/combined.html


### Nudging workflow

A nudging (nudge to fine) workflow template is available and can be run with the
following minimum arguments: `nudging-config`, `reference-restarts`, `initial-condition`,
and `output-url`, e.g., using the example config in `./nudging/examples/argo_clouds_off.yaml`:

    argo submit --from workflowtemplate/nudging \
        -p nudging-config="$(< ./nudging/examples/argo_clouds_off.yaml)" \
        -p reference-restarts="gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts" \
        -p initial-condition="20160801.001500" \
        -p output-url="gs://vcm-ml-scratch/brianh/nudge-to-fine-test" 


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
