# syntax=docker/dockerfile:experimental
FROM ubuntu:18.04 as bld

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && \
    apt-get install -y  --no-install-recommends \
    autoconf \
    bats \
    curl \
    cython3 \
    g++ \
    gcc \
    gfortran \
    git \
    libblas-dev \
    libffi-dev \
    liblapack-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    libpython3-dev \
    libtool \
    libtool-bin \
    m4 \
    make  \
    openssl \
    perl \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    rsync \
    wget

RUN wget -q http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz && \
    tar xzf mpich-3.1.4.tar.gz && \
    cd mpich-3.1.4 && \
    ./configure --enable-fortran --enable-cxx --prefix=/usr --enable-fast=all,O3 && \
    make -j24


RUN cd /mpich-3.1.4 && make install && ldconfig

# Install ESMF
ENV ESMF_DIR=/esmf \
    ESMF_INSTALL_PREFIX=/usr/local/esmf \
    ESMF_INSTALL_MODDIR=/usr/local/esmf/include \
    ESMF_INSTALL_HEADERDIR=/usr/local/esmf/include \
    ESMF_INSTALL_LIBDIR=/usr/local/esmf/lib \
    ESMF_INSTALL_BINDIR=/usr/local/esmf/bin \
    ESMF_NETCDF_INCLUDE=/usr/include \
    ESMF_NETCDF_LIBS="-lnetcdf -lnetcdff" \
    ESMF_BOPT=O3

RUN git clone -b ESMF_8_0_0 --depth 1 https://git.code.sf.net/p/esmf/esmf $ESMF_DIR && \
    cd $ESMF_DIR && \
    make lib -j24 && \
    make install && \
    make installcheck


# INSTALL FMS
ENV CC=/usr/bin/mpicc \
    FC=/usr/bin/mpif90 \
    LDFLAGS="-L/usr/lib" \
    LOG_DRIVER_FLAGS="--comments" \
    CPPFLAGS="-I/usr/include -Duse_LARGEFILE -DMAXFIELDMETHODS_=500 -DGFS_PHYS" \
    FCFLAGS="-fcray-pointer -Waliasing -ffree-line-length-none -fno-range-check -fdefault-real-8 -fdefault-double-8 -fopenmp"

COPY external/fv3gfs-fortran/FMS /FMS
RUN apt-get install -y automake
RUN cd /FMS && autoreconf --install && ./configure && \
    cd /FMS && make -j8 && \
    mv /FMS/*/*.mod /FMS/*/*.o /FMS/*/*.h /FMS/


# download and install NCEP libraries
RUN git config --global http.sslverify false && \
    git clone https://github.com/NCAR/NCEPlibs.git && \
    mkdir /opt/NCEPlibs && \
    cd NCEPlibs && \
    git checkout 3da51e139d5cd731c9fc27f39d88cb4e1328212b && \
    echo "y" | ./make_ncep_libs.sh -s linux -c gnu -d /opt/NCEPlibs -o 1


ENV ESMF_DIR=/usr/local/esmf
ENV FMS_DIR=/FMS
ENV FV3GFS_FORTRAN_DIR=/external/fv3gfs-fortran
ENV ESMF_INC="-I${ESMF_DIR}/include -I${ESMF_DIR}/mod/modO3/Linux.gfortran.64.mpiuni.default/"

ENV FMS_LIB=${FMS_DIR}/libFMS/.libs/
ENV ESMF_LIB=${ESMF_DIR}/lib
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}

# build/install the fortran model
COPY external/fv3gfs-fortran/ /tmp/fortran-build
RUN cd /tmp/fortran-build/FV3 && \
    ./configure gnu_docker && \
    make -j 8 && \
    PREFIX=/usr/local make install

#
# Python Stuff Here
#

# Python components here
COPY docker/prognostic_run/requirements/base.txt /tmp/requirements.txt
COPY constraints.txt /tmp/constraints.txt

# Enforce constraints are used for all pip installs
ENV PIP_CONSTRAINT=/tmp/constraints.txt
# Equivalent to passing --no-cache-dir to every pip install
ENV PIP_NO_CACHE_DIR=off

RUN pip3 install wheel && \
    pip3 install --upgrade pip==20.2.4 setuptools==40.1.0 && \
    ln -s /bin/python3 /bin/python && \
    ln -s /bin/pip3 /bin/pip

# fv3gfs-util
COPY /external/fv3gfs-util /external/fv3gfs-util
RUN pip3 install /external/fv3gfs-util 

COPY /external/fv3gfs-wrapper /fv3gfs-wrapper
RUN pip3 install jinja2 && make -C /fv3gfs-wrapper/lib

# cache external package installation
RUN mkdir -p /fv3net/external /fv3net/workflows && \
    mv /external/* /fv3net/external/ && \
    mv /fv3gfs-wrapper /fv3net/external/fv3gfs-wrapper
ENV MPI=mpich
RUN pip3 install /fv3net/external/fv3gfs-wrapper

COPY external/vcm /fv3net/external/vcm
RUN pip3 install /fv3net/external/vcm

COPY external/loaders /fv3net/external/loaders
RUN pip3 install /fv3net/external/loaders

COPY external/fv3fit /fv3net/external/fv3fit
RUN pip3 install /fv3net/external/fv3fit

COPY external/fv3kube /fv3net/external/fv3kube
RUN pip3 install /fv3net/external/fv3kube

RUN echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

# Install fv3gfs run script
RUN apt-get install -y jq
RUN pip3 install yq==2.11.1
COPY workflows/prognostic_c48_run/runfv3 /usr/local/bin/
RUN chmod 755 /usr/local/bin/runfv3

#Install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk jq python3-dev python3-pip kubectl gfortran


# Install post_process
COPY workflows/post_process_run /fv3net/workflows/post_process_run
RUN pip3 install -c /tmp/constraints.txt -e /fv3net/workflows/post_process_run

# these are needed for "click" to work
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY workflows/prognostic_c48_run/ /fv3net/workflows/prognostic_c48_run
RUN pip install /fv3net/workflows/prognostic_c48_run

ENV PYTHONPATH=/fv3net/workflows/prognostic_c48_run:${PYTHONPATH}
WORKDIR /fv3net/workflows/prognostic_c48_run
CMD ["bash"]

###############################################################################
#  prognostic-run tests
###############################################################################
FROM bld as test

ENV GOOGLE_APPLICATION_CREDENTIALS /tmp/key.json
RUN --mount=type=secret,id=gcp,dst=/tmp/key.json \
    ["bash", "-o", "pipefail", "-c", "pytest | tee /tmp/test-results.txt"]

###############################################################################
#  prognostic-run image
###############################################################################
FROM bld AS prognostic-run
# Copy results from test stage to ensure docker buildkit executes it
COPY --from=test /tmp/test-results.txt .

# build the docs
RUN apt-get install -y graphviz && pip3 install sphinx sphinx_rtd_theme
RUN make -C docs html

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG

###############################################################################
#  notebook image
###############################################################################
FROM bld AS notebook

COPY docker/prognostic_run/requirements/notebook.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt-get install -y bash-completion vim netcdf-bin

RUN useradd --create-home --shell /bin/bash jovyan

USER jovyan
WORKDIR /home/jovyan
VOLUME /home/jovyan

ENTRYPOINT ["jupyter", "lab"]
CMD ["--ip", "0.0.0.0", "--port", "8888", "--no-browser"]