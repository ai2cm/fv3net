# make sure this version works with requirements version
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# An nvidia outage was breaking image builds, and we don't need to install anything
# nvidia specific currently.  Feel free to remove if that changes.
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && \
    apt-get install -y  --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        libmpich-dev \
        libnetcdf-dev \
        libtcmalloc-minimal4 \
        python3-setuptools

# Equivalent to passing --no-cache-dir to every pip install
ENV PIP_NO_CACHE_DIR=off

# Install dependencies
COPY docker/fv3fit_torch/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

COPY external/vcm /fv3net/external/vcm
RUN pip3 install --no-dependencies /fv3net/external/vcm

COPY external/artifacts /fv3net/external/artifacts
RUN pip3 install --no-dependencies /fv3net/external/artifacts

COPY external/loaders /fv3net/external/loaders
RUN pip3 install --no-dependencies /fv3net/external/loaders

COPY external/fv3fit /fv3net/external/fv3fit
RUN pip3 install --no-dependencies /fv3net/external/fv3fit

# overwrite install cuda dgl
RUN pip3 install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html

# Greatly improves memory performance of tensorflow training
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG
