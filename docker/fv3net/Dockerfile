FROM ubuntu:18.04

# install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -


RUN apt-get update && apt-get install -y google-cloud-sdk jq python3-dev python3-pip kubectl gfortran

COPY docker/fv3net/requirements.txt /tmp/requirements.txt
RUN pip3 install wheel && pip3 install -r /tmp/requirements.txt

# cache external package installation
COPY external/fv3config /fv3net/external/fv3config
COPY external/vcm /fv3net/external/vcm
RUN pip3 install -e /fv3net/external/vcm -e /fv3net/external/fv3config

COPY . /fv3net
RUN pip3 install --no-deps -e /fv3net
ENV PYTHONPATH=/fv3net/external/runtime:${PYTHONPATH}

WORKDIR /fv3net
