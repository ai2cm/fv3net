ARG MAMBAFORGE_VERSION=4.14.0-0
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 AS base

ARG PNETCDF_VERSION=1.12.3
ENV PNETCDF_VERSION=${PNETCDF_VERSION}

ARG LIBNETCDF_VERSION=4.8.1
ENV LIBNETCDF_VERSION=${LIBNETCDF_VERSION}

ARG NETCDF_FORTRAN_VERSION=4.6.0
ENV NETCDF_FORTRAN_VERSION=${NETCDF_FORTRAN_VERSION}

ARG ESMF_VERSION=8.4.0
ENV ESMF_VERSION=${ESMF_VERSION}

ARG GCC_VERSION=10.*
ENV GCC_VERSION=${GCC_VERSION}

ENV USER=root
ENV LOGNAME=root

SHELL ["/bin/bash", "-c"]

# install wget
RUN apt-get update && apt-get install -y wget g++

# install miniforge mamba
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    chmod +x Miniforge3-$(uname)-$(uname -m).sh && ./Miniforge3-$(uname)-$(uname -m).sh -b -p "/opt/conda"

ENV PATH "/opt/conda/bin:${PATH}"

# Install common packages
RUN mamba install --yes -c conda-forge \
    cmake \
    make \
    wget \
    curl \
    subversion \
    m4 \
    pytest \
    pytest-cov\
    pyyaml \
    vim \
    openssh && \
    rm -rf /opt/conda/pkgs/*

# Install version locked packages
RUN mamba install --yes -c conda-forge \
    libnetcdf=${LIBNETCDF_VERSION}=*openmpi* \
    netcdf-fortran=${NETCDF_FORTRAN_VERSION}=*openmpi* \
    esmf=${ESMF_VERSION}=*openmpi* \
    gcc_linux-64=${GCC_VERSION} \
    gxx_linux-64=${GCC_VERSION} \
    openmpi-mpifort \
    gfortran_linux-64=${GCC_VERSION} && \
    rm -rf /opt/conda/pkgs/* && \
    ln -sf /opt/conda/bin/x86_64-conda-linux-gnu-ar /opt/conda/bin/ar && \
    ln -sf /opt/conda/bin/x86_64-conda-linux-gnu-ranlib /opt/conda/bin/ranlib

# Install cpan packages
RUN cpan install XML::LibXML Switch

# Build pnetcdf
RUN curl -L -k -o "${PWD}/pnetcdf.tar.gz" \
    https://parallel-netcdf.github.io/Release/pnetcdf-${PNETCDF_VERSION}.tar.gz && \
    mkdir "${PWD}/pnetcdf" && \
    tar -xvf "${PWD}/pnetcdf.tar.gz" -C "${PWD}/pnetcdf" --strip-components=1 && \
    rm -rf "${PWD}/pnetcdf.tar.gz" && \
    cd "${PWD}/pnetcdf" && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    ./configure --prefix /opt/conda --disable-cxx --enable-shared \
    MPICC=/opt/conda/bin/mpicc \
    MPICXX=/opt/conda/bin/mpicxx \
    MPIF77=/opt/conda/bin/mpif77 \
    MPIF90=/opt/conda/bin/mpif90 && \
    make -j4 && \
    make install && \
    rm -rf "${PWD}/pnetcdf"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    curl ca-certificates software-properties-common \
    gcc make libtool libhwloc-dev libx11-dev libxt-dev libedit-dev \
    libical-dev ncurses-dev perl python-dev tcl-dev tk-dev swig libexpat-dev libssl-dev \
    libxext-dev libxft-dev autoconf automake \
    postgresql-12 postgresql-server-dev-all postgresql-contrib \
    expat libedit2 python3 sendmail-bin sudo tcl tk && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y python3.7 python3.7-dev && \
    rm -rf /var/lib/apt/lists/*

# Install additional packages
RUN mamba install --yes -c conda-forge \
    lapack \
    blas && \
    rm -rf /opt/conda/pkgs/*

# Install dependencies
COPY docker/prognostic_scream_run/scream/components/eamxx/docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext git

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

RUN apt-get update && apt-get install -y google-cloud-sdk
RUN gcloud config set project vcm-ml

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV IS_DOCKER=TRUE
COPY docker/prognostic_scream_run/scream/ /src/E3SM

COPY docker/prognostic_scream_run/requirements.in /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY external/vcm /fv3net/external/vcm
RUN pip install --no-dependencies /fv3net/external/vcm

COPY external/artifacts /fv3net/external/artifacts
RUN pip3 install --no-dependencies /fv3net/external/artifacts

COPY external/loaders /fv3net/external/loaders
RUN pip3 install --no-dependencies /fv3net/external/loaders

COPY external/fv3fit /fv3net/external/fv3fit
RUN pip3 install --no-dependencies /fv3net/external/fv3fit

COPY workflows/prognostic_scream_run/ /fv3net/workflows/prognostic_scream_run
RUN pip install -r /fv3net/workflows/prognostic_scream_run/requirements.txt
RUN pip install -e /fv3net/workflows/prognostic_scream_run

# this is a workaround to enforce h5py uses the same hdf5 version as the one used by tensorflow
RUN pip uninstall -y tensorflow h5py
RUN pip install tensorflow==2.8.0

# this is a workaround because python does not recognize the already installed py package
RUN pip uninstall -y py
RUN pip install py==1.11.0

COPY docker/prognostic_scream_run/precompile_scream.sh /src/precompile_scream.sh
ENV CC=/opt/conda/bin/mpicc
ENV CXX=/opt/conda/bin/mpicxx
ENV FC=/opt/conda/bin/mpif90
ENV AR=/opt/conda/bin/x86_64-conda-linux-gnu-ar
ENV RANLIB=/opt/conda/bin/x86_64-conda-linux-gnu-ranlib
ENV LDFLAGS="-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/opt/conda/lib -Wl,-rpath-link,/opt/conda/lib -L/opt/conda/lib"
ENV GOOGLE_APPLICATION_CREDENTIALS /tmp/key.json
RUN --mount=type=secret,id=gcp,dst=/tmp/key.json \
    export FSSPEC_GS_TOKEN=/tmp/key.json && \
    /src/precompile_scream.sh

# If I don't want to mix up library stuff for scream
# I'll probably have to pip install cupy, and then no-dep install
# cupy-xarray since both will be installed previously
RUN mamba install cupy cupy-xarray h5py

COPY projects/scream/working /fv3net/projects/scream/working

COPY projects/scream/build_scream /src/build_scream

WORKDIR /fv3net/projects/scream/working
