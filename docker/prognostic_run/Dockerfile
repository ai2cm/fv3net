# syntax=docker/dockerfile:experimental
ARG BASE_IMAGE
FROM ${BASE_IMAGE} as bld

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

COPY docker/prognostic_run/scripts/install_esmf.sh install_esmf.sh
RUN bash install_esmf.sh /usr/local/esmf

COPY docker/prognostic_run/scripts/install_fms.sh install_fms.sh
COPY external/fv3gfs-fortran/FMS /FMS
RUN bash install_fms.sh /FMS

COPY docker/prognostic_run/scripts/install_nceplibs.sh .
RUN bash install_nceplibs.sh /opt/NCEPlibs


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

# build/install the fortran model
COPY external/fv3gfs-fortran/ /tmp/fortran-build
RUN cd /tmp/fortran-build/FV3 && \
    ./configure gnu_docker && \
    sed -i "33i CALLPYFORT=Y" conf/configure.fv3 && \
    make -j 8 && \
    PREFIX=/usr/local make install

# gcloud

#Install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk jq python3-dev python3-pip kubectl gfortran graphviz


#
# Python Stuff Here
#
# Equivalent to passing --no-cache-dir to every pip install
ENV PIP_NO_CACHE_DIR=off

# Install dependencies
COPY docker/prognostic_run/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY /external/fv3gfs-wrapper /fv3gfs-wrapper
RUN pip3 install jinja2 && make -C /fv3gfs-wrapper/lib

# cache external package installation
RUN mkdir -p /fv3net/external /fv3net/workflows && \
    mv /fv3gfs-wrapper /fv3net/external/fv3gfs-wrapper

ENV MPI=mpich
RUN pip3 install --no-dependencies /fv3net/external/fv3gfs-wrapper

COPY external/vcm /fv3net/external/vcm
RUN pip3 install --no-dependencies /fv3net/external/vcm

COPY external/artifacts /fv3net/external/artifacts
RUN pip3 install --no-dependencies /fv3net/external/artifacts

COPY external/loaders /fv3net/external/loaders
RUN pip3 install --no-dependencies /fv3net/external/loaders

COPY external/fv3fit /fv3net/external/fv3fit
RUN pip3 install --no-dependencies /fv3net/external/fv3fit

COPY external/fv3kube /fv3net/external/fv3kube
RUN pip3 install --no-dependencies /fv3net/external/fv3kube

COPY workflows/post_process_run /fv3net/workflows/post_process_run
RUN pip3 install --no-dependencies -e /fv3net/workflows/post_process_run

RUN echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir


# these are needed for "click" to work
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY workflows/prognostic_c48_run/ /fv3net/workflows/prognostic_c48_run
RUN pip install --no-dependencies /fv3net/workflows/prognostic_c48_run

# Override microphysics emulation
COPY external/emulation /fv3net/external/emulation
RUN pip install --no-dependencies /fv3net/external/emulation
ENV VAR_META_PATH=/fv3net/external/emulation/microphysics_parameter_metadata.yaml
ENV OUTPUT_FREQ_SEC=18000

# Add emulation project scripts
COPY projects/microphysics/scripts /fv3net/projects/microphysics/scripts
RUN chmod +x /fv3net/projects/microphysics/scripts/*
ENV PATH=/fv3net/projects/microphysics/scripts:${PATH}

ENV PYTHONPATH=/fv3net/workflows/prognostic_c48_run:${PYTHONPATH}
WORKDIR /fv3net/workflows/prognostic_c48_run
CMD ["bash"]

###############################################################################
#  prognostic-run image
###############################################################################
FROM bld AS prognostic-run
# Copy results from test stage to ensure docker buildkit executes it

RUN make -C docs html

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG