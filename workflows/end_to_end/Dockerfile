FROM python:3.7.7-buster

# install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

RUN apt-get update && apt-get install -y google-cloud-sdk jq kubectl gfortran

COPY docker/fv3net/requirements.txt /tmp/requirements.txt
RUN pip install wheel && pip3 install -r /tmp/requirements.txt

RUN mkdir -p /fv3net
# cache external package installation

COPY external/fv3config /fv3net/external/fv3config
COPY external/vcm /fv3net/external/vcm
COPY external/vcm /fv3net/external/runtime
RUN pip install -e /fv3net/external/vcm -e /fv3net/external/fv3config


WORKDIR /fv3net

COPY fv3net fv3net
COPY workflows workflows
COPY setup.py setup.py

RUN pip install --no-deps -e /fv3net
ENV PYTHONPATH=/fv3net/external/runtime:${PYTHONPATH}

