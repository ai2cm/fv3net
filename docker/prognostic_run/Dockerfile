FROM us.gcr.io/vcm-ml/fv3gfs-python:v0.4.3

COPY docker/prognostic_run/requirements.txt /tmp/requirements.txt
RUN pip3 install wheel && pip3 install -r /tmp/requirements.txt

# cache external package installation
COPY external/fv3config /fv3net/external/fv3config
COPY external/vcm /fv3net/external/vcm
COPY external/loaders /fv3net/external/loaders
RUN pip3 install -e /fv3net/external/vcm -e /fv3net/external/fv3config -e /fv3net/external/loaders

COPY . /fv3net
ENV PYTHONPATH=/fv3net/external/runtime:/fv3net:${PYTHONPATH}
