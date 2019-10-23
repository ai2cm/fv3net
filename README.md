fv3net
==============================

Improving the GFDL FV3 model physics with machine learning

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `2019-07-10-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

# Setting up the environment

This project uses an anaconda environment file to specify the computational environment. To build this environment run
	
    conda env create -n fv3net environment.yml

Then, in the future this will need to be loaded with
	
    conda activate fv3net


To make sure that `src` is in the python path, you can run

    python setup.py develop

or possible add it to the `PYTHONPATH` environmental variable.

# Pre-processing

To run the pre-processing---regridding to 1 degree and computing advection
using semi-lagrangian scheme---run 
	
    make data

This command operates with data in the Vulcan google cloud storage bucket
`vcm-ml-data`. You should already have access if you are using a VM that we
provisioned for you. Otherwise you will need to login with [`gcloud auth
application-default`](https://cloud.google.com/sdk/gcloud/reference/auth/application-default/).

# Opening Data

After downloading the data and pre-processing, it can be opened from python
by running
```python
from src.data import open_dataset
ds = open_dataset('1degTrain')
```
This dataset contains the apparent heating (Q1) and moistening (Q2** and many potential input variables.

# Extending this code

## Adding new model types

*This interface is a work in progress. It might be better to define a class
based interface. Let's defer that to when we have more than one model type*

To add a new model type one needs to create two new files
```
src.models.{model_type}/train.py
src.models.{model_type}/test.py
```

The training script should take in a yaml file of options with this command line interface
```
python -m src.models.{model_type}.train --options options.yaml output.pkl
```

# Deploying on k8s

Make docker image for this workflow and push it to GCR

    make push_image

Create a K8S cluster:

    bash provision_cluster.sh

This cluster has some big-mem nodes for doing the FV3 run, which requires at least a n1-standard-2 VM for 
C48.

Install argo following [these instructions](https://github.com/argoproj/argo/blob/master/demo.md).

Submit an argo job using

    argo submit --watch argo-fv3net.yml



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
