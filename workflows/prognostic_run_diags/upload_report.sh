set -x

gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
jupyter nbconvert --execute combined.ipynb
gsutil cp combined.html $OUTPUT
