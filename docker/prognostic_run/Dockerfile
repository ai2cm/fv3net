# syntax=docker/dockerfile:experimental
ARG BASE_IMAGE
FROM ${BASE_IMAGE} as bld

# build/install the fortran model
COPY external/fv3gfs-fortran/ /tmp/fortran-build
COPY .environment-scripts/install_fv3gfs_fortran.sh .
RUN CALLPYFORT=Y bash install_fv3gfs_fortran.sh /tmp/fortran-build/FV3 gnu_docker /usr/local

# Install the Python dependencies
COPY docker/prognostic_run/requirements.txt /fv3net/docker/prognostic_run/requirements.txt
COPY external/vcm /fv3net/external/vcm
COPY external/artifacts /fv3net/external/artifacts
COPY external/loaders /fv3net/external/loaders
COPY external/fv3fit /fv3net/external/fv3fit
COPY external/fv3kube /fv3net/external/fv3kube
COPY workflows/post_process_run /fv3net/workflows/post_process_run
COPY workflows/prognostic_c48_run/ /fv3net/workflows/prognostic_c48_run
COPY external/emulation /fv3net/external/emulation
COPY projects/microphysics/scripts /fv3net/projects/microphysics/scripts
COPY external/radiation /fv3net/external/radiation
COPY .environment-scripts/install_fv3net_python_dependencies.sh .
RUN bash install_fv3net_python_dependencies.sh /fv3net

# compile and install the wrapper. Then import fv3gfs.wrapper as a minimal test
# that it works.
RUN make -C /tmp/fortran-build/FV3 wrapper_build &&\
    pip3 install --no-dependencies /tmp/fortran-build/FV3/wrapper/dist/*.whl && \
    python3 -c 'import fv3gfs.wrapper'

RUN echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir


# these are needed for "click" to work
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Override microphysics emulation
ENV VAR_META_PATH=/fv3net/external/emulation/microphysics_parameter_metadata.yaml
ENV OUTPUT_FREQ_SEC=18000

# Add emulation project scripts
COPY projects/microphysics/scripts /fv3net/projects/microphysics/scripts
RUN chmod +x /fv3net/projects/microphysics/scripts/*
ENV PATH=/fv3net/projects/microphysics/scripts:${PATH}

ENV PYTHONPATH=/fv3net/workflows/prognostic_c48_run:/fv3net/external/fv3fit:/fv3net/external/emulation:/fv3net/external/vcm:/fv3net/external/artifacts:/fv3net/external/loaders:/fv3net/external/fv3kube:/fv3net/workflows/post_process_run:/fv3net/external/radiation:${PYTHONPATH}
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