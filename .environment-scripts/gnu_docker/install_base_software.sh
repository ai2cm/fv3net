#!/bin/bash
set -e

# Note that if this script is modified the base image will need to be rebuilt.
CLONE_PREFIX=$1
INSTALL_PREFIX=$2
CONDA_ENV=$3
SCRIPTS=$4


apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && \
    apt-get install -y  --no-install-recommends \
    autoconf \
    automake \
    bats \
    cmake \
    curl \
    cython3 \
    g++ \
    gcc \
    gfortran \
    git \
    libblas-dev \
    libffi-dev \
    liblapack-dev \
    libmpich-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    libpython3-dev \
    libtool \
    libtool-bin \
    m4 \
    make  \
    mpich \
    openssl \
    perl \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-cffi \
    rsync \
    wget

apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk jq python3-dev python3-pip kubectl gfortran graphviz
