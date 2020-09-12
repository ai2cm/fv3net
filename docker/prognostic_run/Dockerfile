ARG FORTRAN_IMAGE

FROM ${FORTRAN_IMAGE} AS fv3gfs-fortran-build

FROM ubuntu:19.10 AS fv3gfs-environment

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gcc \
    git \
    libblas-dev \
    liblapack-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    perl \
    make \
    rsync \
    libffi-dev \
    openssl \
    libopenmpi3 \
    bats

COPY --from=fv3gfs-fortran-build /NCEPlibs /NCEPlibs

# install NCEP libraries
RUN cd /NCEPlibs && \
    mkdir /opt/NCEPlibs && \
    echo "y" | ./make_ncep_libs.sh -s linux -c gnu -d /opt/NCEPlibs -o 1

## Pull FMS and ESMF
##---------------------------------------------------------------------------------
# these images are defined in fv3gfs-fortran
# KEEP UPDATED IN BOTH DOCKERFILES
FROM us.gcr.io/vcm-ml/fms-build@sha256:868e79a8ef4921f655a6f1fdd32c4eaa200d1157b076a4f85587689bb892e64c AS fv3gfs-fms
FROM us.gcr.io/vcm-ml/esmf-build@sha256:d3154afc4f4b57f9c253be1d84c405b4a3ac508eebbfe5cd0a8c91f65a8287be AS fv3gfs-esmf

#-----------------------------------------------------------------------------
FROM ubuntu:19.10 as fv3config-inputdata

RUN apt-get update && \
    apt-get install -y python3 python3-pip wget && \
    pip3 install --no-cache-dir gsutil
RUN mkdir -p /fv3config-cache/gs/vcm-fv3config && \
    gsutil -m cp -r gs://vcm-fv3config/ /fv3config-cache/gs && \
    rm -rf /fv3config-cache/gs/vcm-fv3config/data/base_forcing/v1.0 \
        /fv3config-cache/gs/vcm-fv3config/data/orographic_data/v1.0/C384 \
        /fv3config-cache/gs/vcm-fv3config/data/orographic_data/v1.0/C96 \
        /fv3config-cache/orographic_data/C384 \
        /fv3config-cache/base_forcing

#-----------------------------------------------------------------------------
FROM fv3gfs-environment as fv3gfs-wrapper-env

COPY /external/fv3gfs-wrapper/requirements.txt /fv3gfs-wrapper/requirements.txt
# TODO add constraints here

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libpython3-dev \
    libopenmpi3 \
    python3-dev \
    python3-setuptools \
    python3-pip \
    cython3 && \
    pip3 install wheel && \
    ln -s /bin/python3 /bin/python && \
    pip3 install -r /fv3gfs-wrapper/requirements.txt && \
    ln -s /bin/pip3 /bin/pip

#-----------------------------------------------------------------------------
FROM fv3gfs-wrapper-env AS fv3gfs-wrapper-build

ENV FMS_DIR=/usr/local/FMS
ENV ESMF_DIR=/usr/local/esmf
ENV FV3GFS_FORTRAN_DIR=/external/fv3gfs-fortran
ENV ESMF_INC="-I${ESMF_DIR}/include -I${ESMF_DIR}/mod/modO3/Linux.gfortran.64.mpiuni.default/"

ENV FMS_LIB=${FMS_DIR}/libFMS/.libs/
ENV ESMF_LIB=${ESMF_DIR}/lib/libO3/Linux.gfortran.64.mpiuni.default/
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}

ARG FV3GFS_BUILD_ROOT=/

COPY --from=fv3gfs-fms /FMS $FMS_DIR
COPY --from=fv3gfs-esmf /usr/local/esmf ${ESMF_DIR}

COPY --from=fv3gfs-fms /FMS/libFMS/.libs/*.so* /lib64/
COPY --from=fv3gfs-esmf /usr/local/esmf/lib/libO3/Linux.gfortran.64.mpiuni.default/*.so* /lib64/

# copy wrapper and fortran sources
# directory copy commands must be separate due to docker limitations
COPY /external/fv3gfs-wrapper/lib/Makefile /external/fv3gfs-wrapper/lib/coupler_lib.F90 /fv3gfs-wrapper/lib/
COPY --from=fv3gfs-fortran-build ${FV3GFS_BUILD_ROOT}/stochastic_physics ${FV3GFS_FORTRAN_DIR}/stochastic_physics
COPY --from=fv3gfs-fortran-build ${FV3GFS_BUILD_ROOT}/FV3 ${FV3GFS_FORTRAN_DIR}/FV3

RUN PREFIX=/usr/local make -C ${FV3GFS_FORTRAN_DIR}/FV3 install

COPY /external/fv3gfs-wrapper/templates /fv3gfs-wrapper/templates
COPY /external/fv3gfs-wrapper/examples /fv3gfs-wrapper/examples
COPY /external/fv3gfs-wrapper/docs /fv3gfs-wrapper/docs
# files can be copied in one line
COPY /external/fv3gfs-wrapper/fv3gfs/wrapper/dynamics_properties.json \
    /external/fv3gfs-wrapper/fv3gfs/wrapper/physics_properties.json \
    /fv3gfs-wrapper/fv3gfs/wrapper/
COPY /external/fv3gfs-wrapper/fill_templates.py \
    /external/fv3gfs-wrapper/HISTORY.md \
    /external/fv3gfs-wrapper/LICENSE \
    /external/fv3gfs-wrapper/Makefile \
    /external/fv3gfs-wrapper/MANIFEST.in \
    /external/fv3gfs-wrapper/README.md \
    /external/fv3gfs-wrapper/setup.cfg \
    /external/fv3gfs-wrapper/setup.py \
    /fv3gfs-wrapper/
COPY /external/fv3gfs-wrapper/fv3gfs /fv3gfs-wrapper/fv3gfs

RUN make -C /fv3gfs-wrapper build

#-----------------------------------------------------------------------------
FROM fv3gfs-wrapper-build AS wrapper

# copy and install dependency packages
COPY /external/fv3config /external/fv3config
COPY /external/fv3gfs-util /external/fv3gfs-util

# cache model data
ENV FV3CONFIG_CACHE_DIR=/inputdata
COPY --from=fv3config-inputdata /fv3config-cache $FV3CONFIG_CACHE_DIR/fv3config-cache

RUN chmod -R 777 $FV3CONFIG_CACHE_DIR && \
    echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

# copy the tests
COPY /external/fv3gfs-wrapper/tests /fv3gfs-wrapper/tests

# interactive shell by default
CMD ["bash"]

FROM wrapper AS bld

COPY docker/prognostic_run/requirements.txt /tmp/requirements.txt
COPY constraints.txt /tmp/constraints.txt
RUN pip3 install wheel && pip3 install --upgrade pip && pip3 install -c /tmp/constraints.txt -r /tmp/requirements.txt

# cache external package installation
RUN mkdir -p /fv3net/external /fv3net/workflows && \
    mv /external/* /fv3net/external/ && \
    mv /fv3gfs-wrapper /fv3net/external/fv3gfs-wrapper

COPY external/fv3fit /fv3net/external/fv3fit
COPY external/fv3kube /fv3net/external/fv3kube
COPY external/vcm /fv3net/external/vcm
COPY external/loaders /fv3net/external/loaders

COPY workflows/prognostic_c48_run/ /fv3net/workflows/prognostic_c48_run

COPY docker/prognostic_run/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# install the packages
RUN /entrypoint.sh

ENV PYTHONPATH=/fv3net/workflows/prognostic_c48_run:${PYTHONPATH}
WORKDIR /fv3net/workflows/prognostic_c48_run
ENTRYPOINT [ "/entrypoint.sh" ]

FROM bld AS test

RUN pytest
