# Use identical base image (20.04) to what we use in the fv3net prognostic run:
FROM ubuntu@sha256:9101220a875cee98b016668342c489ff0674f247f6ca20dfc91b91c0f28581ae

ENV FV3NET_DIR=/fv3net
ENV FV3NET_SCRIPTS=${FV3NET_DIR}/.environment-scripts

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
ENV SHiELD_AVX_LEVEL=-mavx

ENV SUBMODULE_DIR=/SHiELD
COPY external/SHiELD-wrapper/submodules ${SUBMODULE_DIR}/

# Build FMS, NCEPlibs, and SHiELD using SHiELD_build with position independent
# code.  FMS_CPPDEFS is needed to address
# https://github.com/NOAA-GFDL/FMS/issues/426
RUN cd ${SUBMODULE_DIR}/SHiELD_build/Build && \
    FMS_CPPDEFS="-DHAVE_GETTID" \
    FC=${SHiELD_FC} \
    CC=${SHiELD_CC} \
    CXX=${SHiELD_CXX} \
    LD=${SHiELD_LD} \
    TEMPLATE=site/gnu.mk \
    AVX_LEVEL=${SHiELD_AVX_LEVEL} \
    ./COMPILE shield repro 64bit gnu pic

# Install the Python requirements for the prognostic run.
COPY docker/prognostic_run/requirements.txt ${FV3NET_DIR}/docker/prognostic_run/requirements.txt
RUN CC=${SHiELD_CC} \
    MPICC=${SHiELD_CC} \
    pip install --no-cache-dir -r ${FV3NET_DIR}/docker/prognostic_run/requirements.txt

# Prescribe LDSHARED as well to ensure that the wrapper is built with the GNU 10
# compilers.  See https://github.com/ai2cm/fv3gfs-fortran/issues/330 for more
# comprehensive discussion.  This may not be absolutely necessary (we do not
# worry about this for the FV3GFS prognostic run), but it feels cleaner not to
# mix compilers if we can avoid it.
COPY external/SHiELD-wrapper/wrapper ${FV3NET_DIR}/external/SHiELD-wrapper/wrapper
RUN make -C ${FV3NET_DIR}/external/SHiELD-wrapper/wrapper \
    FC=${SHiELD_FC} \
    CC=${SHiELD_CC} \
    CXX=${SHiELD_CXX} \
    LD=${SHiELD_LD} \
    LDSHARED="${SHiELD_CC} -shared" \
    AVX_LEVEL=${SHiELD_AVX_LEVEL} \
    PIC=Y \
    OPENMP=Y \
    AVX=Y \
    REPRO=Y \
    build install
RUN python3 -c "import shield.wrapper"

# Install the fv3net packages.  Do this last, because these packages change the most
# frequently during our development.  This allows us to get the most out of caching
# the prior build steps.
COPY .environment-scripts/install_fv3net_packages.sh ${FV3NET_SCRIPTS}/
COPY external/vcm ${FV3NET_DIR}/external/vcm
COPY external/artifacts ${FV3NET_DIR}/external/artifacts
COPY external/loaders ${FV3NET_DIR}/external/loaders
COPY external/fv3fit ${FV3NET_DIR}/external/fv3fit
COPY external/fv3kube ${FV3NET_DIR}/external/fv3kube
COPY workflows/post_process_run ${FV3NET_DIR}/workflows/post_process_run
COPY workflows/prognostic_c48_run/ ${FV3NET_DIR}/workflows/prognostic_c48_run
COPY external/emulation ${FV3NET_DIR}/external/emulation
COPY external/radiation ${FV3NET_DIR}/external/radiation
RUN bash ${FV3NET_SCRIPTS}/install_fv3net_packages.sh \
    ${FV3NET_DIR}/external/vcm \
    ${FV3NET_DIR}/external/artifacts \
    ${FV3NET_DIR}/external/loaders \
    ${FV3NET_DIR}/external/fv3fit \
    ${FV3NET_DIR}/external/fv3kube \
    ${FV3NET_DIR}/workflows/post_process_run \
    ${FV3NET_DIR}/workflows/prognostic_c48_run \
    ${FV3NET_DIR}/external/emulation \
    ${FV3NET_DIR}/external/radiation

RUN echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

# these are needed for "click" to work
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Override microphysics emulation
ENV VAR_META_PATH=${FV3NET_DIR}/external/emulation/microphysics_parameter_metadata.yaml
ENV OUTPUT_FREQ_SEC=18000

# Add fv3net packages to the PYTHONPATH
ENV PYTHONPATH=${FV3NET_DIR}/workflows/prognostic_c48_run:${FV3NET_DIR}/external/fv3fit:${FV3NET_DIR}/external/emulation:${FV3NET_DIR}/external/vcm:${FV3NET_DIR}/external/artifacts:${FV3NET_DIR}/external/loaders:${FV3NET_DIR}/external/fv3kube:${FV3NET_DIR}/workflows/post_process_run:${FV3NET_DIR}/external/radiation:${PYTHONPATH}

# Set DGLBACKEND to pytorch to silence warnings that it is unset; if unset it is
# set to pytorch, so this just makes it explicit.
ENV DGLBACKEND=pytorch

WORKDIR ${FV3NET_DIR}/workflows/prognostic_c48_run
CMD ["bash"]
