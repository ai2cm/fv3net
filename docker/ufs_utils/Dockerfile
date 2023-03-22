FROM ubuntu:20.04 as bld

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    gfortran \
    g++ \
    git \
    libxerces-c-dev \
    make \
    python3 \
    apt-transport-https \
    ca-certificates \
    gnupg \
    gettext

# Install gcloud
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update && apt-get install -y google-cloud-sdk
RUN gcloud config set project vcm-ml

# Install UFS_UTILS dependencies
COPY docker/ufs_utils/spack.yaml .
RUN git clone -c feature.manyFiles=true https://github.com/spack/spack.git --branch v0.19.1
RUN . spack/share/spack/setup-env.sh && \
    spack env create ufs-utils-env spack.yaml && \
    spack env activate ufs-utils-env && \
    spack concretize && \
    spack install --fail-fast && \
    spack clean -a

# Install and test UFS_UTILS
RUN git clone https://github.com/ufs-community/UFS_UTILS.git --branch ufs_utils_1_9_0 && \
    cd UFS_UTILS && \
    git submodule update --init --recursive && \
    cd ..
RUN . spack/share/spack/setup-env.sh && \
    spack env activate ufs-utils-env && \
    export CC=mpicc && \
    export FC=mpif90 && \
    cd UFS_UTILS && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=../install .. && \
    make -j2 VERBOSE=1 && \
    ctest --verbose --output-on-failure --rerun-failed && \
    make install

# Add script for running argo workflow
COPY docker/ufs_utils/restart_files_to_nggps_initial_condition.sh /local-tools/bin/
ENV PATH="${PATH}:/local-tools/bin"
