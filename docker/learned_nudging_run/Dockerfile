FROM us.gcr.io/vcm-ml/fv3gfs-python:v0.4.0


COPY docker/learned_nudging_run/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# cache external package installation
COPY external/fv3config /fv3net/external/fv3config
COPY external/vcm /fv3net/external/vcm
RUN pip3 install -e /fv3net/external/vcm -e /fv3net/external/fv3config

COPY . /fv3net
RUN pip3 install --no-deps -e /fv3net
