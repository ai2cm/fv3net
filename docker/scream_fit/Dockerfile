FROM ubuntu:20.04

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
COPY docker/scream_fit/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY external/vcm /fv3net/external/vcm
RUN pip3 install --no-dependencies /fv3net/external/vcm

COPY external/artifacts /fv3net/external/artifacts
RUN pip3 install --no-dependencies /fv3net/external/artifacts

COPY external/loaders /fv3net/external/loaders
RUN pip3 install --no-dependencies /fv3net/external/loaders

COPY external/fv3fit /fv3net/external/fv3fit
RUN pip3 install --no-dependencies /fv3net/external/fv3fit

# Greatly improves memory performance of tensorflow training
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.8/dist-packages/nvidia/cuda_runtime/lib
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pytest /fv3net/external/fv3fit/tests/training/test_main.py /fv3net/external/fv3fit/tests/training/test_train.py /fv3net/external/fv3fit/tests/test_io.py