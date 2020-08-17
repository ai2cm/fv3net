## Dataflow

Dataflow jobs run in a "serverless" style where data is piped between workers who
execute a single function.  This workflow type is good for pure python operations
that are easily parallelizable.  Dataflow handles resource provision and worker
scaling making it a somewhat straightforward option.

For a dataflow pipeline, jobs can be tested locally using a DirectRunner, e.g., 

    python -m fv3net.pipelines.extract_tars test_tars test_output_dir --runner DirectRunner

Remote dataflow jobs should be submitted with the `dataflow.sh` script in
this directory. See `./dataflow.sh -h` for more details on how to use this
script.

Other forms of dataflow submission are not reproducible, and are strongly
discouraged. This script is tested by CI and handles the difficult task of
bundling VCM private packages, and specifying all the combined dependencies
that are not pre-installed in dataflow workers (see [this
webpage](https://cloud.google.com/dataflow/docs/concepts/sdk-worker-dependencies)).


### Troubleshooting

If you get an error `Could not create workflow; user does not have write
access to project` upon trying to submit the dataflow job, do `gcloud auth
application-default login` first and then retry.
