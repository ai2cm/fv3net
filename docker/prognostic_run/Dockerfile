# syntax=docker/dockerfile:experimental
ARG BASE_IMAGE
FROM ${BASE_IMAGE} as bld

COPY .environment-scripts /fv3net/.environment-scripts

COPY external/fv3gfs-fortran/ /tmp/fortran-build
COPY docker/prognostic_run/requirements.txt /fv3net/docker/prognostic_run/requirements.txt
COPY external/vcm /fv3net/external/vcm
COPY external/artifacts /fv3net/external/artifacts
COPY external/loaders /fv3net/external/loaders
COPY external/fv3fit /fv3net/external/fv3fit
COPY external/fv3kube /fv3net/external/fv3kube
COPY workflows/post_process_run /fv3net/workflows/post_process_run
COPY workflows/prognostic_c48_run/ /fv3net/workflows/prognostic_c48_run
COPY external/emulation /fv3net/external/emulation
COPY external/radiation /fv3net/external/radiation

RUN bash /fv3net/.environment-scripts/setup_environment.sh \
    fv3net \
    gnu_docker \
    / \
    /usr/local \
    /fv3net \
    /FMS \
    /tmp/fortran-build/FV3 \
    Y

RUN python3 -c 'import fv3gfs.wrapper'

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