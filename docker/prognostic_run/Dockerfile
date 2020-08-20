FROM us.gcr.io/vcm-ml/fv3gfs-python:52a80fa838d737652ba6b8c950b983ab032872f9 AS bld

COPY docker/prognostic_run/requirements.txt /tmp/requirements.txt
RUN pip3 install wheel && pip3 install --upgrade pip && pip3 install -r /tmp/requirements.txt

# cache external package installation
RUN mkdir -p /fv3net/external /fv3net/workflows

COPY external/fv3config /fv3net/external/fv3config
COPY external/fv3fit /fv3net/external/fv3fit
COPY external/vcm /fv3net/external/vcm
COPY external/loaders /fv3net/external/loaders

COPY workflows/prognostic_c48_run/ /fv3net/workflows/prognostic_c48_run

COPY docker/prognostic_run/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# install the packages
RUN /entrypoint.sh

ENV PYTHONPATH=/fv3net/workflows/prognostic_c48_run:${PYTHONPATH}
WORKDIR /fv3net/workflows/prognostic_c48_run
ENTRYPOINT [ "/entrypoint.sh" ]


FROM bld AS test

RUN pytest


FROM bld
