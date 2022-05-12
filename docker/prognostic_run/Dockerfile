# syntax=docker/dockerfile:experimental
ARG BASE_IMAGE
FROM ${BASE_IMAGE} as bld

# build/install the fortran model
COPY external/fv3gfs-fortran/ /tmp/fortran-build
RUN cd /tmp/fortran-build/FV3 && \
    ./configure gnu_docker && \
    sed -i "33i CALLPYFORT=Y" conf/configure.fv3 && \
    make -j 8 && \
    PREFIX=/usr/local make install

# Python Stuff Here
#
# Equivalent to passing --no-cache-dir to every pip install
ENV PIP_NO_CACHE_DIR=off

# Install dependencies
COPY docker/prognostic_run/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# compile and install the wrapper. Then import fv3gfs.wrapper as a minimal test
# that it works.
RUN make -C /tmp/fortran-build/FV3 wrapper_build &&\
    pip3 install --no-dependencies /tmp/fortran-build/FV3/wrapper/dist/*.whl && \
    python3 -c 'import fv3gfs.wrapper'

COPY external/vcm /fv3net/external/vcm
RUN pip3 install --no-dependencies /fv3net/external/vcm

COPY external/artifacts /fv3net/external/artifacts
RUN pip3 install --no-dependencies /fv3net/external/artifacts

COPY external/loaders /fv3net/external/loaders
RUN pip3 install --no-dependencies /fv3net/external/loaders

COPY external/fv3fit /fv3net/external/fv3fit
RUN pip3 install --no-dependencies /fv3net/external/fv3fit

COPY external/fv3kube /fv3net/external/fv3kube
RUN pip3 install --no-dependencies /fv3net/external/fv3kube

COPY workflows/post_process_run /fv3net/workflows/post_process_run
RUN pip3 install --no-dependencies -e /fv3net/workflows/post_process_run

RUN echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir


# these are needed for "click" to work
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY workflows/prognostic_c48_run/ /fv3net/workflows/prognostic_c48_run
RUN pip install --no-dependencies /fv3net/workflows/prognostic_c48_run

# Override microphysics emulation
COPY external/emulation /fv3net/external/emulation
RUN pip install --no-dependencies /fv3net/external/emulation
ENV VAR_META_PATH=/fv3net/external/emulation/microphysics_parameter_metadata.yaml
ENV OUTPUT_FREQ_SEC=18000

# Add emulation project scripts
COPY projects/microphysics/scripts /fv3net/projects/microphysics/scripts
RUN chmod +x /fv3net/projects/microphysics/scripts/*
ENV PATH=/fv3net/projects/microphysics/scripts:${PATH}

ENV PYTHONPATH=/fv3net/workflows/prognostic_c48_run:${PYTHONPATH}
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