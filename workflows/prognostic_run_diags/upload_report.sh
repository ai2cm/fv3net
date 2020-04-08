set -x

gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
jupyter nbconvert --execute combined.ipynb
gsutil cp combined.html gs://vcm-ml-public/experiments-2020-03/prognostic_run_diags/combined.html
