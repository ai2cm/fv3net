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
COPY docker/torch/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

COPY external/vcm /fv3net/external/vcm
RUN pip3 install --no-dependencies /fv3net/external/vcm

COPY external/fv3fit /fv3net/external/fv3fit
RUN pip3 install --no-dependencies /fv3net/external/fv3fit

# install torch
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

# Add FME project scripts
COPY projects/full_model_emulation /fv3net/projects/full_model_emulation
RUN chmod +x /fv3net/projects/full_model_emulation/scripts/*
ENV PATH=/fv3net/projects/full_model_emulation/torch:${PATH}

# Greatly improves memory performance of tensorflow training
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG
