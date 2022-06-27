FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

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
        python3-setuptools \
        wget \
        jq

# Install yq
RUN wget https://github.com/mikefarah/yq/releases/download/v4.25.3/yq_linux_amd64.tar.gz -O - | \
  tar xz && mv yq_linux_amd64 /usr/bin/yq

# Equivalent to passing --no-cache-dir to every pip install
ENV PIP_NO_CACHE_DIR=off


# Install dependencies
COPY docker/fv3fit/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY external/vcm /fv3net/external/vcm
RUN pip3 install --no-dependencies /fv3net/external/vcm

COPY external/artifacts /fv3net/external/artifacts
RUN pip3 install --no-dependencies /fv3net/external/artifacts

COPY external/loaders /fv3net/external/loaders
RUN pip3 install --no-dependencies /fv3net/external/loaders

COPY external/fv3fit /fv3net/external/fv3fit
RUN pip3 install --no-dependencies /fv3net/external/fv3fit

COPY external/emulation /fv3net/external/emulation
RUN pip3 install --no-dependencies /fv3net/external/emulation


# Add emulation project scripts
COPY projects/microphysics/scripts /fv3net/projects/microphysics/scripts
RUN chmod +x /fv3net/projects/microphysics/scripts/*
ENV PATH=/fv3net/projects/microphysics/scripts:${PATH}

# Greatly improves memory performance of tensorflow training
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG
