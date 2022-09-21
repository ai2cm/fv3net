# syntax=docker/dockerfile:experimental
ARG BASE_IMAGE
FROM ${BASE_IMAGE} as bld

COPY .environment-scripts/setup_environment_post_base.sh ${FV3NET_SCRIPTS}/

# Copy data and files required for building the fortran model
COPY external/fv3gfs-fortran/ ${FV3GFS_FORTRAN_DIR}
COPY .environment-scripts/install_fv3gfs_fortran.sh ${FV3NET_SCRIPTS}/
COPY .environment-scripts/${PLATFORM}/install_fv3gfs_fortran.sh ${FV3NET_SCRIPTS}/${PLATFORM}/
RUN bash /fv3net/.environment-scripts/setup_environment.sh \
    fv3gfs-fortran \
    ${PLATFORM} \
    ${CLONE_PREFIX} \
    ${INSTALL_PREFIX} \
    ${FV3NET_DIR} \
    ${CALLPYFORT}

# Copy data and files required for creating the Python environment
COPY .environment-scripts/install_python_requirements.sh ${FV3NET_SCRIPTS}/
COPY .environment-scripts/${PLATFORM}/install_python_requirements.sh ${FV3NET_SCRIPTS}/${PLATFORM}/
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
COPY .environment-scripts/${PLATFORM}/install_python_wrapper.sh ${FV3NET_SCRIPTS}/${PLATFORM}
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

# Add emulation project scripts and script for post-build steps
COPY .environment-scripts/post_build_steps.sh ${FV3NET_SCRIPTS}/
COPY projects/microphysics/scripts /fv3net/projects/microphysics/scripts
RUN bash /fv3net/.environment-scripts/setup_environment.sh \
    post-build \
    ${PLATFORM} \
    ${CLONE_PREFIX} \
    ${INSTALL_PREFIX} \
    ${FV3NET_DIR} \
    ${CALLPYFORT}

RUN echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

COPY .environment-scripts/environment_variables.sh ${FV3NET_SCRIPTS}/

WORKDIR /fv3net/workflows/prognostic_c48_run
COPY .environment-scripts/gnu_docker/activate_environment.sh ${FV3NET_SCRIPTS}/${PLATFORM}/
RUN chmod +x ${FV3NET_SCRIPTS}/${PLATFORM}/activate_environment.sh
ENTRYPOINT ["/fv3net/.environment-scripts/gnu_docker/activate_environment.sh"]
CMD = ["bash"]

###############################################################################
#  prognostic-run image
###############################################################################
FROM bld AS prognostic-run
# Copy results from test stage to ensure docker buildkit executes it

RUN make -C docs html

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG
