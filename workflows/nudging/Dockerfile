ARG MODEL_IMAGE

FROM us.gcr.io/vcm-ml/fv3gfs-python:v0.3.0 as cache_data

# keep credentials in a separate image from final image, so they won't be uploaded
# if we distribute the final image
# also use a separate data image so it doesn't re-download when MODEL_IMAGE is updated
ENV GOOGLE_APPLICATION_CREDENTIALS=/gcs_key.json
ARG GCS_CREDENTIALS
RUN test -n "$GCS_CREDENTIALS"  # must explicitly test to avoid silent failure
COPY $GCS_CREDENTIALS $GOOGLE_APPLICATION_CREDENTIALS

# Run a script that writes the run directory so we cache the underlying data files
# This is unnecessary if only running this workflow once, but saves having to
# re-download the data every time the model is run.
COPY cache_data.py fv3config.yml /tmp/cache_data/
RUN cd /tmp/cache_data && python3 cache_data.py

FROM $MODEL_IMAGE as nudging

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY --from=cache_data $FV3CONFIG_CACHE_DIR/fv3config-cache/gs $FV3CONFIG_CACHE_DIR/fv3config-cache/gs

# new subdirectories need to have write permission for all users
RUN chmod -R 777 $FV3CONFIG_CACHE_DIR
