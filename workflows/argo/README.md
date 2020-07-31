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


This command will make output like this:

    Name:                prognostic-run-diags-sps8h
    Namespace:           default
    ServiceAccount:      default
    Status:              Pending
    Created:             Thu Apr 30 12:01:17 -0700 (now)


This job can be monitored by running

    argo watch <Name>

Moreover, the templates within this workflows can be used by other workflows.


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
argo submit --from workflowtemplate/prognostic-run-report \
    -p runs="$(< rundirs.json)" \
    -p docker-image=<dockerimage> \
    --name <name>
```

If succesful, the completed report will be available at
`gs://vcm-ml-public/argo/<name>/index.html`. This can be accessed from a web browser using this link:

    http://storage.googleapis.com/vcm-ml-public/argo/<name>/index.html 


[1]: http://storage.googleapis.com/vcm-ml-public/experiments-2020-03/prognostic_run_diags/combined.html
