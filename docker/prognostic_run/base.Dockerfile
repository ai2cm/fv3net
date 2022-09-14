# syntax=docker/dockerfile:experimental

# If you edit this file, you must upload a new version by incrementing
# the PROGNOSTIC_BASE_VERSION in /Makefile and running
# `make push_image_prognostic_run_base push_image_prognostic_run_base_gpu`

ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS prognostic-run-base

# An nvidia outage was breaking image builds, and we don't need to install anything
# nvidia specific currently.  Feel free to remove if that changes.
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

COPY .environment-scripts /fv3net/.environment-scripts
COPY external/fv3gfs-fortran/FMS /FMS
RUN bash /fv3net/.environment-scripts/setup_environment.sh \
    base \
    gnu_docker \
    / \
    /usr/local \
    /fv3net \
    /FMS \
    /tmp/fortran-build \
    Y

# I think we can clean these up at some point, but for now they are OK.
ENV ESMF_DIR=/usr/local/esmf
ENV CALLPY_DIR=/usr/local
ENV FMS_DIR=/FMS
ENV FV3GFS_FORTRAN_DIR=/external/fv3gfs-fortran

ENV FMS_LIB=${FMS_DIR}/libFMS/.libs/
ENV ESMF_LIB=${ESMF_DIR}/lib
ENV CALLPYFORT_LIB=${CALLPY_DIR}/lib
ENV CALLPYFORT_INCL=${CALLPY_DIR}/include
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}:${CALLPYFORT_LIB}

ENV CALL_PY_FORT_DIR=/call_py_fort 
ENV PYTHONPATH=${CALL_PY_FORT_DIR}/src/:$PYTHONPATH