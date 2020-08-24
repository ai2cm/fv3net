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
template. The template can also be called directly from the command line:

```
argo submit --from workflowtemplate/run-fv3gfs \
    -p fv3config="$(< fv3config.yaml)" \
    -p runfile="$(< runfile.py)" \
    -p output-url=<gcs-output-url> \
    -p fv3gfs-image=<fv3gfs-docker-image> \ 
    -p post-process-image=<post-process-docker-image> \
    -p chunks="$(< chunks.yaml)" \
    -p cpu="6" \
    -p memory="8Gi" \
    -p submission-count="2"
```
The `chunks`, `cpu`, `memory` and `submission-count` arguments are optional. Defaults are defined 
in the workflow template.

#### Running multiple segments

The workflow will submit `submission-count` model segments in sequence. The diagnostic 
outputs from each segment will automatically be appended to the previous segment's at
`output-url`. All other outputs (restart files, logging, etc.) will be saved to
`output-url/artifacts/{timestamp}` where `timestamp` corresponds to the start time of
each segment. The duration of each segment is defined by the `fv3config` object passed
to the workflow.

#### Post-processing and chunking

The post-processing can convert netCDF diagnostic outputs of the form `name.tile?.nc`
to zarr with user-specified chunks and rechunk zarrs output by fv3gfs-wrapper. To
specify that a set of netCDF outputs should be converted to zarr, their chunking must be
given in the provided `chunks.yaml`. See post-processing script at
`workflows/post_process_run` for more details.

WARNING: if `submission-count` is greater than 1, the chunk size in time must evenly
divide the length of the time dimension for each diagnostic output.

### Prognostic run report

The `prognostic-run-diags` workflow template will generate reports for
prognostic runs. See this [example][1].

|Parameter| Description|
|-------- |-------------|
| runs | A json-encoded list of {"name": ..., "url": ...} items |
| docker-image | The docker image to use |

- `runs`: If `runs` is `""`, then all the timesteps will be processed.

The outputs will be stored at the directory
`gs://vcm-ml-public/argo/<workflow name>`, where `<workflow name>` is NOT the
name of the workflow template, but of the created `workflow` resource.

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


If succesful, the completed report will be available at
`gs://vcm-ml-public/argo/<name>/index.html`. This can be accessed from a web browser using this link:

    http://storage.googleapis.com/vcm-ml-public/argo/<name>/index.html 


[1]: http://storage.googleapis.com/vcm-ml-public/experiments-2020-03/prognostic_run_diags/combined.html
