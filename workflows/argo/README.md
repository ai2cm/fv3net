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
    -p docker-image=<dockerimage> \
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


### Running fv3gfs with argo

The `run-fv3gfs` template is a general purpose workflow to do fv3gfs simulations on the
cloud. It does post-processing on the fly and the workflow can run the model in
sequential segments to increase reliability and reduce the memory requirement for
the post-processing step. See the nudging workflow at
`workflows/argo/nudging/nudging.yaml` for an example usage of the `run-fv3gfs`
template.

| Parameter            | Description                                                                                           |
|----------------------|-------------------------------------------------------------------------------------------------------|
| fv3config            | String representation of an fv3config object                                                          |
| runfile              | String representation of an fv3gfs runfile                                                            |
| output-url           | GCS url for outputs                                                                                   |
| fv3gfs-image         | Docker image used to run model. Currently only `us.gcr.io/vcm-ml/prognostic_run` supported.           |
| post-process-image   | Docker image used to post-process and upload outputs                                                  |
| chunks               | (optional) String describing desired chunking of diagnostics                                          |
| cpu                  | (optional) Requested cpu for run-model step                                                           |
| memory               | (optional) Requested memory for run-model step                                                        |
| segment-count        | (optional) Number of segments to run                                                                  |
| working-volume-name  | (optional) Name of volume for temporary work. Volume claim must be made prior to run-fv3gfs workflow. |
| external-volume-name | (optional) Name of volume with external data. E.g. for restart data in a nudged run.                  |

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
defined in the given `chunks` parameter. For example:
```yaml
atmos_8xdaily.zarr:
  time: 8
nudging_tendencies.zarr:
  time: 1
sfc_dt_atmos.zarr:
  time: 96
```

Some diagnostics have default chunking. See post-processing script at 
`workflows/post_process_run/post_process.py` for more details.

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
| image-tag             | Tag for fv3net, prognostic-run, and post_process_run images                |
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
| store-true-args       | (optional) String of store-true flags for prognostic run prepare_config.py |
| chunks                | (optional) Custom dimension rechunking mapping for prognostic run outputs  |
| segment-count         | (optional) Number of prognostic run segments; default 1                    |
| cpu-prog              | (optional) Number of cpus for prognostic run nodes; default 6              |
| memory-prog           | (optional) Memory for prognostic run nodes; default 6Gi                    |
| work-volume-name      | (optional) Working volume name, prognostic run; default 'work-volume'      |
| external-volume-name  | (optional) External volume name, prognostic run; default 'external-volume' |


### Prognostic run report

The `prognostic-run-diags` workflow template will generate reports for
prognostic runs. See this [example][1].

| Parameter    | Description                                                  |
|--------------|--------------------------------------------------------------|
| runs         | A json-encoded list of {"name": ..., "url": ...} items       |
| docker-image | The docker image to use                                      |
| make-movies  | (optional) whether to generate movies. Defaults to false     |
| flags        | (optional) flags to pass to save_prognostic_diags.py script. |

The outputs will be stored at the directory
`gs://vcm-ml-public/argo/<workflow name>`, where `<workflow name>` is NOT the
name of the workflow template, but of the created `workflow` resource.

To specify what verification data use when computing the diagnostics, use the `--verification`
flag. E.g. specifying the argo parameter `flags="--verification nudged_c48_fv3gfs_2016` will use a
year-long nudged-to-obs C48 run as verification. By default, the `40day_may2020` simulation
is used as verification (see fv3net catalog).

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
    -p docker-image=<dockerimage> \
    --name <name>
```


If successful, the completed report will be available at
`gs://vcm-ml-public/argo/<name>/index.html`. This can be accessed from a web browser using this link:

    http://storage.googleapis.com/vcm-ml-public/argo/<name>/index.html 

If you wish to generate movies of column-integrated heating and moistening along with the report, 
add the parameter `-p make-movies="true"`. By default, the movies will not be created.

[1]: http://storage.googleapis.com/vcm-ml-public/experiments-2020-03/prognostic_run_diags/combined.html
