# syntax=docker/dockerfile:experimental

# If you edit this file, you must upload a new version by incrementing
# the PROGNOSTIC_BASE_VERSION in /Makefile and running
# `make push_image_prognostic_run_base push_image_prognostic_run_base_gpu`

ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS prognostic-run-base

# An nvidia outage was breaking image builds, and we don't need to install anything
# nvidia specific currently.  Feel free to remove if that changes.
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

ENV FV3GFS_FORTRAN_DIR=/tmp/fortran-build
ENV CLONE_PREFIX=/
ENV INSTALL_PREFIX=/usr/local
ENV PLATFORM=gnu_docker
ENV FV3NET_DIR=/fv3net
ENV FMS_DIR=/FMS
ENV FV3_DIR=${FV3GFS_FORTRAN_DIR}/FV3
ENV CALLPYFORT=Y

ENV FV3NET_SCRIPTS=${FV3NET_DIR}/.environment-scripts

# Copy general scripts
COPY .environment-scripts/setup_environment.sh ${FV3NET_SCRIPTS}/
COPY .environment-scripts/install_nceplibs.sh ${FV3NET_SCRIPTS}/
COPY .environment-scripts/install_esmf.sh ${FV3NET_SCRIPTS}/
COPY .environment-scripts/install_fms.sh ${FV3NET_SCRIPTS}/
COPY .environment-scripts/install_call_py_fort.sh ${FV3NET_SCRIPTS}/

# Copy platform-specific scripts
COPY .environment-scripts/${PLATFORM}/install_base_software.sh ${FV3NET_SCRIPTS}/${PLATFORM}/
COPY .environment-scripts/${PLATFORM}/install_nceplibs.sh ${FV3NET_SCRIPTS}/${PLATFORM}/
COPY .environment-scripts/${PLATFORM}/install_esmf.sh ${FV3NET_SCRIPTS}/${PLATFORM}/
COPY .environment-scripts/${PLATFORM}/install_fms.sh ${FV3NET_SCRIPTS}/${PLATFORM}/
COPY .environment-scripts/${PLATFORM}/install_call_py_fort.sh ${FV3NET_SCRIPTS}/${PLATFORM}/


COPY external/fv3gfs-fortran/FMS ${FMS_DIR}
RUN bash ${FV3NET_SCRIPTS}/setup_environment.sh \
    base \
    ${PLATFORM} \
    ${CLONE_PREFIX} \
    ${INSTALL_PREFIX} \
    ${FV3NET_DIR} \
    ${FMS_DIR} \
    ${FV3_DIR} \
    ${CALLPYFORT}

ENV ESMF_DIR=/usr/local/esmf
ENV CALLPY_DIR=/usr/local

ENV FMS_LIB=${FMS_DIR}/libFMS/.libs/
ENV ESMF_LIB=${ESMF_DIR}/lib
ENV CALLPYFORT_LIB=${CALLPY_DIR}/lib
ENV CALLPYFORT_INCL=${CALLPY_DIR}/include
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}:${CALLPYFORT_LIB}

ENV CALL_PY_FORT_DIR=/call_py_fort 
ENV PYTHONPATH=${CALL_PY_FORT_DIR}/src/:$PYTHONPATH
