# syntax=docker/dockerfile:experimental
ARG BASE_IMAGE
FROM ${BASE_IMAGE} as bld

COPY .environment-scripts/setup_development_environment.sh ${FV3NET_SCRIPTS}/
COPY .environment-scripts/${PLATFORM}/development_configuration_variables.sh ${FV3NET_SCRIPTS}/${PLATFORM}/

# Copy data and files required for building the fortran model
COPY external/fv3gfs-fortran/ ${FV3GFS_FORTRAN_DIR}
COPY .environment-scripts/install_fv3gfs_fortran.sh ${FV3NET_SCRIPTS}/
RUN bash /fv3net/.environment-scripts/setup_environment.sh \
    fv3gfs-fortran \
    ${PLATFORM} \
    ${CLONE_PREFIX} \
    ${INSTALL_PREFIX} \
    ${FV3NET_DIR} \
    ${CALLPYFORT}

# Copy requirements file needed for creating the Python environment
COPY docker/prognostic_run/requirements.txt /fv3net/docker/prognostic_run/requirements.txt
RUN bash /fv3net/.environment-scripts/setup_environment.sh \
    python-requirements \
    ${PLATFORM} \
    ${CLONE_PREFIX} \
    ${INSTALL_PREFIX} \
    ${FV3NET_DIR} \
    ${CALLPYFORT}

# Copy data and files required for building the Python wrapper
COPY .environment-scripts/install_python_wrapper.sh ${FV3NET_SCRIPTS}/
RUN bash /fv3net/.environment-scripts/setup_environment.sh \
    wrapper \
    ${PLATFORM} \
    ${CLONE_PREFIX} \
    ${INSTALL_PREFIX} \
    ${FV3NET_DIR} \
    ${CALLPYFORT}
RUN python3 -c 'import fv3gfs.wrapper'

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
RUN bash /fv3net/.environment-scripts/setup_environment.sh \
    fv3net-packages \
    ${PLATFORM} \
    ${CLONE_PREFIX} \
    ${INSTALL_PREFIX} \
    ${FV3NET_DIR} \
    ${CALLPYFORT}

# Add emulation project scripts
COPY projects/microphysics/scripts /fv3net/projects/microphysics/scripts

RUN echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

# these are needed for "click" to work
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Override microphysics emulation
ENV VAR_META_PATH=$FV3NET_DIR/external/emulation/microphysics_parameter_metadata.yaml
ENV OUTPUT_FREQ_SEC=18000

# Add emulation project scripts
ENV PATH=$FV3NET_DIR/projects/microphysics/scripts:${PATH}

# Add fv3net packages to the PYTHONPATH
ENV PYTHONPATH=$FV3NET_DIR/workflows/prognostic_c48_run:$FV3NET_DIR/external/fv3fit:$FV3NET_DIR/external/emulation:$FV3NET_DIR/external/vcm:/fv3net/external/artifacts:$FV3NET_DIR/external/loaders:$FV3NET_DIR/external/fv3kube:$FV3NET_DIR/workflows/post_process_run:$FV3NET_DIR/external/radiation:${PYTHONPATH}

WORKDIR /fv3net/workflows/prognostic_c48_run
CMD = ["bash"]

###############################################################################
#  prognostic-run image
###############################################################################
FROM bld AS prognostic-run
# Copy results from test stage to ensure docker buildkit executes it

RUN make -C docs html

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG
