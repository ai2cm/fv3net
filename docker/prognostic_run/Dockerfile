FROM us.gcr.io/vcm-ml/fv3gfs-python:v0.3.1

COPY docker/prognostic_run/requirements.txt /tmp/requirements.txt
RUN pip3 install wheel && pip3 install -r /tmp/requirements.txt

# cache external package installation
COPY external/fv3config /fv3net/external/fv3config
RUN pip3 install -e /fv3net/external/fv3config

COPY . /fv3net
RUN pip3 install --no-deps -e /fv3net
ENV PYTHONPATH=/fv3net/external/runtime:${PYTHONPATH}
