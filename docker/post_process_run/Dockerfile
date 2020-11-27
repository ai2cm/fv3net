FROM ubuntu:20.04

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    git \
    libblas-dev \
    liblapack-dev \
    libnetcdf-dev \
    netcdf-bin \
    libnetcdff-dev \
    libmpich-dev \
    autoconf \
    perl \
    make \
    rsync \
    python3-dev \
    python3-pip

# install FRE-NCtools including fregrid
COPY external/FRE-NCtools /FRE-NCtools
RUN cd FRE-NCtools && export CFLAGS=-Wno-traditional NETCDF_LDFLAGS=-lm && \
    \autoreconf -i && ./configure && make && make install

# install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

RUN apt-get update && apt-get install -y google-cloud-sdk
RUN gcloud config set project vcm-ml

# install python dependencies
RUN pip3 install pytest
COPY workflows/post_process_run/requirements.txt /tmp/requirements.txt
COPY constraints.txt /tmp/constraints.txt
RUN pip3 install -c /tmp/constraints.txt -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt /tmp/constraints.txt

# install python package
COPY workflows/post_process_run /post_process_run
RUN pip3 install /post_process_run --no-deps --no-cache-dir

# run tests
RUN cd /post_process_run && pytest -vv tests/test_fregrid.py && rm -rf /post_process_run

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG
