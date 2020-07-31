FROM us.gcr.io/vcm-ml/fv3gfs-python:v0.5.0

COPY docker/prognostic_run/requirements.txt /tmp/requirements.txt
RUN pip3 install wheel && pip3 install --upgrade pip && pip3 install -r /tmp/requirements.txt

# cache external package installation
COPY external/fv3config /external/fv3config
COPY external/fv3fit /external/fv3fit
COPY external/vcm /external/vcm
COPY external/loaders /external/loaders

COPY workflows/prognostic_c48_run/ /workdir


COPY docker/prognostic_run/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/workdir:${PYTHONPATH}
WORKDIR /workdir
ENTRYPOINT [ "/entrypoint.sh" ]


