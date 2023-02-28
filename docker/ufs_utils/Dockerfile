ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS bld

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

COPY restart_files_to_nggps_initial_condition.sh /local-tools/bin/
ENV PATH="${PATH}:/local-tools/bin"
