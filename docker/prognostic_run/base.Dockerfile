# syntax=docker/dockerfile:experimental

# If you edit this file, you must upload a new version by incrementing
# the PROGNOSTIC_BASE_VERSION in /Makefile and running
# `make push_image_prognostic_run_base push_image_prognostic_run_base_gpu`

ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS prognostic-run-base

# An nvidia outage was breaking image builds, and we don't need to install anything
# nvidia specific currently.  Feel free to remove if that changes.
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && \
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

COPY environments/scripts/install-esmf.sh .
RUN bash install-esmf.sh /esmf /usr/local/esmf Linux gfortran default

COPY environments/scripts/install-fms.sh .
COPY external/fv3gfs-fortran/FMS /FMS
ENV FMS_LDFLAGS='-L/usr/lib'
ENV FMS_LOG_DRIVER_FLAGS='--comments'
ENV FMS_CPPFLAGS='-I/usr/include -Duse_LARGEFILE -DMAXFIELDMETHODS_=500 -DGFS_PHYS'
ENV FMS_FCFLAGS='-fcray-pointer -Waliasing -ffree-line-length-none -fno-range-check -fdefault-real-8 -fdefault-double-8 -fopenmp'
ENV FMS_FLAGS="LDFLAGS=$FMS_LDFLAGS LOG_DRIVER_FLAGS=$FMS_LOG_DRIVER_FLAGS CPPFLAGS=$FMS_CPPFLAGS FCFLAGS=$FMS_FCFLAGS"
RUN CC=/usr/bin/mpicc FC=/usr/bin/mpif90 bash install-fms.sh /FMS $FMS_FLAGS

COPY environments/scripts/install-nceplibs.sh .
RUN bash install-nceplibs.sh /NCEPlibs /opt/NCEPlibs linux gnu


ENV ESMF_DIR=/usr/local/esmf
ENV CALLPY_DIR=/usr/local
ENV FMS_DIR=/FMS
ENV FV3GFS_FORTRAN_DIR=/external/fv3gfs-fortran
ENV ESMF_INC="-I${ESMF_DIR}/include -I${ESMF_DIR}/mod/modO3/Linux.gfortran.64.mpiuni.default/"

ENV FMS_LIB=${FMS_DIR}/libFMS/.libs/
ENV ESMF_LIB=${ESMF_DIR}/lib
ENV CALLPYFORT_LIB=${CALLPY_DIR}/lib
ENV CALLPYFORT_INCL=${CALLPY_DIR}/include
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}:${CALLPYFORT_LIB}

RUN cd /opt && git clone https://github.com/nbren12/call_py_fort.git --branch=v0.2.0
ENV CALLPY=/opt/call_py_fort \
    PYTHONPATH=${CALLPY}/src/:$PYTHONPATH
RUN cd ${CALLPY} && make && make install && ldconfig

# Install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk jq python3-dev python3-pip kubectl gfortran graphviz
