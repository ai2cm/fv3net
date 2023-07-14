# Install dependencies

Currently we only support conda environemnts. Install dependencies via

```
conda env create -f environment.yaml
pip install -r requirements.txt
```

Then activate the environment
```
conda activate vsr
```

# Getting access to data

Currently, we load the data dynamically from a Google Cloud Storage bucket on each training run. To get access from a non-Google Cloud Platform (GCP) machine, we have found that you sometimes need to run.

```
gcloud auth application-default login
```

