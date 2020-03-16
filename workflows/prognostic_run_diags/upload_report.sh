set -x

[ -f $GOOGLE_APPLICATION_CREDENTIALS ] && gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
jupyter nbconvert --execute combined.ipynb
gsutil cp combined.html gs://vcm-ml-public/testing-2020-02/prognostic_run_diags/combined.html
