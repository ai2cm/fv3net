FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

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

# to solve memeory leak issue with tensorflow
# https://github.com/tensorflow/tensorflow/issues/34942#issuecomment-783766895
ENV LD_PRELOAD=/path/to/libtcmalloc_minimal.so.4