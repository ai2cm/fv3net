fv3net
==============================
[![CircleCI](https://circleci.com/gh/VulcanClimateModeling/fv3net.svg?style=svg&circle-token=98ccddae8375060a2fbbf240407dd4135d3dcf68)](https://circleci.com/gh/VulcanClimateModeling/fv3net)

TODO: you can see the docs at <link>

Improving the GFDL FV3 model physics with machine learning


## Running fv3gfs with Kubernetes

Docker images with the python-wrapped model and fv3run are available from the
[fv3gfs-python](https://github.com/VulcanClimateModeling/fv3gfs-python) repo.
Kubernetes jobs can be written to run the model on these docker images. A super simple
job would be to perform an `fv3run` command (provided by the
[fv3config package](https://github.com/VulcanClimateModeling/fv3config))
using google cloud storage locations. For example, running the basic model using a
fv3config dictionary in a yaml file to output to a google cloud storage bucket
would look like:

```
fv3run gs://my_bucket/my_config.yml gs://my_bucket/my_outdir
```

If you have a python model runfile you want to execute in place of the default model
script, you could use it by adding e.g. `--runfile gs://my-bucket/my_runfile.py`
to the `fv3run` command.

You could create a kubernetes yaml file which runs such a command on a
`fv3gfs-python` docker image, and submit it manually. However, `fv3config` also
supplies a `run_kubernetes` function to do this for you. See the
[`fv3config`](https://github.com/VulcanClimateModeling/fv3config) documentation for
more complete details.

The basic structure of the command is

    fv3config.run_kubernetes(
        config_location,
        outdir,
        docker_image,
        gcp_secret='gcp_key',
    )

Where `config_location` is a google cloud storage location of a yaml file containing
a fv3config dictionary, outdir is a google cloud storage location to put the resulting
run directory, `docker_image` is the name of a docker image containing `fv3config`
and `fv3gfs-python`, and `gcp_secret` is the name of the secret containing the google
cloud platform access key (as a json file called `key.json`). For our VCM group this
should be set to 'gcp_key'. Additional arguments are
available for configuring the kubernetes job and documented in the `run_kubernetes`
docstring.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
