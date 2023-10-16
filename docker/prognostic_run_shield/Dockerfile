# Use identical base image (20.04) to what we use in the fv3net prognostic run:
# https://github.com/ai2cm/fv3net/blob/382409f76fd26d70457b349a48fa0e63f4f26303/Makefile#L12
#
# This is convenient for a couple reasons:
#
# * It installs Python 3.8 as the system Python, which should allow us to
#   seamlessly install the additional Python packages used in the existing
#   prognostic_run image, to their exact versions.
#
# * It installs a version of MPICH that does not enable link time optimization
#   (LTO) by default.  Newer Ubuntu LTS base images, e.g. 22.04, install
#   versions of MPICH that enable link time optimization by default in their
#   compiler wrappers (i.e. mpif90 and the like; see the presence of -flto=auto
#   and other LTO-related flags in the output of `mpif90 -show`).  LTO can lead
#   to differences in answers between applications that were linked differently
#   (e.g. the fortran executable and the Python-wrapped model). Compiling
#   without LTO allows the fortran executable and the Python wrapper to
#   robustly produce bitwise identical results.  See
#   https://github.com/ai2cm/SHiELD-minimal/pull/6 for some details on MPICH in
#   the Ubuntu LTS 22.04 image.  This can be addressed by manually adding
#   `-fno-lto` compiler flags to Makefile templates, but it is easiest not to 
#   need to worry about this.
FROM ubuntu@sha256:9101220a875cee98b016668342c489ff0674f247f6ca20dfc91b91c0f28581ae

# CALLPYFORT, PLATFORM, CLONE_PREFIX, and INSTALL_PREFIX are not used for
# anything meaningful here.
ENV CALLPYFORT=N
ENV PLATFORM=gnu_docker
ENV CLONE_PREFIX=/
ENV INSTALL_PREFIX=/usr/local
ENV FV3NET_DIR=/fv3net
ENV FV3NET_SCRIPTS=${FV3NET_DIR}/.environment-scripts
COPY .environment-scripts/setup_environment.sh ${FV3NET_SCRIPTS}/

# Install and use GNU version 10 compiliers instead of the default version 9,
# since the latest version of FMS fails to build with GNU version 9
# compilers.
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install \
    cmake \
    gcc-10 \
    gfortran-10 \
    g++-10 \
    git \
    libmpich-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    m4 \
    make \
    mpich \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools

# GNU version 9 compilers are unavoidably installed by default; using
# update-alternatives ensures that gcc, gfortran, and g++ point to GNU version
# 10 compilers, including within MPI-wrapped compilers.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 1
RUN update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-10 1
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 1

# Install google-cloud-sdk and a few other packages based on the original
# prognostic run image.
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl gettext && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk jq python3-dev python3-pip kubectl gfortran graphviz

ENV SHiELD_FC=mpif90
ENV SHiELD_CC=mpicc
ENV SHiELD_CXX=mpicxx
ENV SHiELD_LD=mpif90

ENV SUBMODULE_DIR=/SHiELD
COPY external/SHiELD-wrapper/submodules ${SUBMODULE_DIR}/

# Build FMS, NCEPlibs, and SHiELD using SHiELD_build with position independent
# code FMS_CPPDEFS is needed to address
# https://github.com/NOAA-GFDL/FMS/issues/426
RUN cd ${SUBMODULE_DIR}/SHiELD_build/Build && \
    FMS_CPPDEFS="-DHAVE_GETTID" \
    FC=${SHiELD_FC} \
    CC=${SHiELD_CC} \
    CXX=${SHiELD_CXX} \
    LD=${SHiELD_LD} \
    TEMPLATE=site/gnu.mk \
    AVX_LEVEL=-march=native \
    ./COMPILE shield 64bit gnu pic

COPY docker/prognostic_run/requirements.txt /fv3net/docker/prognostic_run/requirements.txt
COPY .environment-scripts/setup_development_environment.sh ${FV3NET_SCRIPTS}/
COPY .environment-scripts/${PLATFORM}/configuration_variables.sh ${FV3NET_SCRIPTS}/${PLATFORM}/
RUN bash ${FV3NET_SCRIPTS}/setup_environment.sh \
    python-requirements \
    ${PLATFORM} \
    ${CLONE_PREFIX} \
    ${INSTALL_PREFIX} \
    ${FV3NET_DIR} \
    ${CALLPYFORT}

# Prescribe LDSHARED as well to ensure that the wrapper is built with the GNU
# 10 compilers.  See https://github.com/ai2cm/fv3gfs-fortran/issues/330 for
# more comprehensive discussion.  This may not be absolutely necessary (we do
# not worry about this in fv3net), but it feels cleaner not to mix compilers if
# we can avoid it.
COPY external/SHiELD-wrapper/wrapper /wrapper
RUN make -C wrapper \
    FC=${SHiELD_FC} \
    CC=${SHiELD_CC} \
    CXX=${SHiELD_CXX} \
    LD=${SHiELD_LD} \
    LDSHARED="${SHiELD_CC} -shared" \
    PIC=Y \
    OPENMP=Y \
    AVX=Y \
    build install
RUN python3 -c "import shield.wrapper"

# Install the fv3net packages.  Do this last, because these packages change the most
# frequently during our development.  This allows us to get the most out of caching
# the prior build steps.
COPY .environment-scripts/install_fv3net_packages.sh ${FV3NET_SCRIPTS}/
COPY external/vcm /fv3net/external/vcm
COPY external/artifacts /fv3net/external/artifacts
COPY external/loaders /fv3net/external/loaders
COPY external/fv3fit /fv3net/external/fv3fit
COPY external/fv3kube /fv3net/external/fv3kube
COPY workflows/post_process_run /fv3net/workflows/post_process_run
COPY workflows/prognostic_c48_run/ /fv3net/workflows/prognostic_c48_run
COPY external/emulation /fv3net/external/emulation
COPY external/radiation /fv3net/external/radiation
RUN bash ${FV3NET_SCRIPTS}/setup_environment.sh \
    fv3net-packages \
    ${PLATFORM} \
    ${CLONE_PREFIX} \
    ${INSTALL_PREFIX} \
    ${FV3NET_DIR} \
    ${CALLPYFORT}

# Add emulation project scripts
COPY projects/microphysics/scripts /fv3net/projects/microphysics/scripts
ENV PATH=$FV3NET_DIR/projects/microphysics/scripts:${PATH}

RUN echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

# these are needed for "click" to work
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Override microphysics emulation
ENV VAR_META_PATH=$FV3NET_DIR/external/emulation/microphysics_parameter_metadata.yaml
ENV OUTPUT_FREQ_SEC=18000

# Add fv3net packages to the PYTHONPATH
ENV PYTHONPATH=$FV3NET_DIR/workflows/prognostic_c48_run:$FV3NET_DIR/external/fv3fit:$FV3NET_DIR/external/emulation:$FV3NET_DIR/external/vcm:/fv3net/external/artifacts:$FV3NET_DIR/external/loaders:$FV3NET_DIR/external/fv3kube:$FV3NET_DIR/workflows/post_process_run:$FV3NET_DIR/external/radiation:${PYTHONPATH}

WORKDIR /fv3net/workflows/prognostic_c48_run
CMD ["bash"]