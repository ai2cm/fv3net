ARG FMS_IMAGE=fv3gfs-fms-build
ARG ESMF_IMAGE=fv3gfs-esmf-build
ARG NCEPLIBS_IMAGE=fv3gfs-ncep-build

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


# download and install NCEP libraries
RUN git config --global http.sslverify false && \
    git clone https://github.com/NCAR/NCEPlibs.git && \
    mkdir /opt/NCEPlibs && \
    cd NCEPlibs && \
    git checkout 3da51e139d5cd731c9fc27f39d88cb4e1328212b && \
    echo "y" | ./make_ncep_libs.sh -s linux -c gnu -d /opt/NCEPlibs -o 1

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


## Build FMS
##---------------------------------------------------------------------------------
FROM fv3gfs-environment AS fv3gfs-fms-build

ENV CC=mpicc
ENV FC=mpifort
ENV LDFLAGS="-L/usr/lib"
ENV LOG_DRIVER_FLAGS="--comments"
ENV CPPFLAGS="-I/usr/include -Duse_LARGEFILE -DMAXFIELDMETHODS_=500 -DGFS_PHYS"
ENV FCFLAGS="-fcray-pointer -Waliasing -ffree-line-length-none -fno-range-check -fdefault-real-8 -fdefault-double-8 -fopenmp"

COPY /external/fv3gfs-fortran/FMS /FMS
RUN cd /FMS && autoreconf --install && ./configure && \
    cd /FMS && make -j8 && \
    mv /FMS/*/*.mod /FMS/*/*.o /FMS/*/*.h /FMS/

## Build ESMF
##---------------------------------------------------------------------------------
FROM fv3gfs-environment AS fv3gfs-esmf-build

ENV ESMF_DIR=/esmf
ENV ESMF_INSTALL_PREFIX=/usr/local/esmf
ENV ESMF_NETCDF_INCLUDE=/usr/include
ENV ESMF_NETCDF_LIBS="-lnetcdf -lnetcdff"
ENV ESMF_BOPT=O3

RUN git clone -b ESMF_8_0_0 --depth 1 https://git.code.sf.net/p/esmf/esmf $ESMF_DIR && \
    cd $ESMF_DIR && \
    make lib -j8 && \
    make install && \
    make installcheck

FROM $FMS_IMAGE AS fv3gfs-fms
FROM $ESMF_IMAGE AS fv3gfs-esmf

## Build FV3 executable in its own image
##---------------------------------------------------------------------------------
FROM fv3gfs-environment AS fv3gfs-fortran-build

ARG compile_option
ARG configure_file=configure.fv3.gnu_docker

COPY /external/fv3gfs-fortran/stochastic_physics /stochastic_physics
COPY /external/fv3gfs-fortran/FV3/coarse_graining /FV3/coarse_graining
COPY /external/fv3gfs-fortran/FV3/conf /FV3/conf
COPY /external/fv3gfs-fortran/FV3/ccpp /FV3/ccpp
COPY /external/fv3gfs-fortran/FV3/cpl /FV3/cpl
COPY /external/fv3gfs-fortran/FV3/gfsphysics /FV3/gfsphysics
COPY /external/fv3gfs-fortran/FV3/io /FV3/io
COPY /external/fv3gfs-fortran/FV3/ipd /FV3/ipd
COPY /external/fv3gfs-fortran/FV3/stochastic_physics /FV3/stochastic_physics
COPY /external/fv3gfs-fortran/FV3/makefile \
    /external/fv3gfs-fortran/FV3/mkDepends.pl \
    /external/fv3gfs-fortran/FV3/atmos_model.F90 \
    /external/fv3gfs-fortran/FV3/LICENSE.md \
    /external/fv3gfs-fortran/FV3/coupler_main.F90 \
    /external/fv3gfs-fortran/FV3/fv3_cap.F90 \
    /external/fv3gfs-fortran/FV3/module_fcst_grid_comp.F90 \
    /external/fv3gfs-fortran/FV3/module_fv3_config.F90 \
    /external/fv3gfs-fortran/FV3/time_utils.F90 \
    /FV3/

# copy appropriate configuration file to configure.fv3
RUN cp /FV3/conf/$configure_file \
        /FV3/conf/configure.fv3 && \
    if [ ! -z $compile_option ]; then sed -i "33i $compile_option" \
        /FV3/conf/configure.fv3; fi

COPY /external/fv3gfs-fortran/FV3/atmos_cubed_sphere /FV3/atmos_cubed_sphere

ENV FMS_DIR=/FMS \
    ESMF_DIR=/usr/local/esmf

ENV ESMF_INC="-I/usr/local/esmf/include -I${ESMF_DIR}/mod/modO3/Linux.gfortran.64.mpiuni.default/" \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_DIR}/lib/libO3/Linux.gfortran.64.mpiuni.default/:${FMS_DIR}/libFMS/.libs/

COPY --from=fv3gfs-fms /FMS $FMS_DIR
COPY --from=fv3gfs-esmf /usr/local/esmf $ESMF_DIR

COPY /external/fv3gfs-fortran/FV3 /FV3
COPY /external/fv3gfs-fortran/stochastic_physics /stochastic_physics

RUN cd /FV3 && make clean_no_dycore && make libs_no_dycore -j8

COPY /external/fv3gfs-fortran/FV3/atmos_cubed_sphere /FV3/atmos_cubed_sphere

RUN cd /FV3/atmos_cubed_sphere && make clean && cd /FV3 && make -j8

#-----------------------------------------------------------------------------
FROM fv3gfs-environment as fv3gfs-wrapper-env

COPY /external/fv3gfs-wrapper/requirements.txt /fv3gfs-wrapper/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libpython3-dev \
    libopenmpi3 \
    python3-dev \
    python3-setuptools \
    python3-pip \
    cython3 && \
    pip3 install wheel && \
    pip3 install -r /fv3gfs-wrapper/requirements.txt  && \
    ln -s /bin/python3 /bin/python && \
    ln -s /bin/pip3 /bin/pip

#-----------------------------------------------------------------------------
FROM fv3gfs-wrapper-env AS fv3gfs-wrapper-build

ENV FMS_DIR=/fv3gfs-wrapper/lib/external/FMS
ENV ESMF_DIR=/fv3gfs-wrapper/lib/external/esmf
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
COPY --from=fv3gfs-fortran-build ${FV3GFS_BUILD_ROOT}/stochastic_physics /fv3gfs-wrapper/lib/external/stochastic_physics
COPY --from=fv3gfs-fortran-build ${FV3GFS_BUILD_ROOT}/FV3 /fv3gfs-wrapper/lib/external/FV3

RUN ls /fv3gfs-wrapper/lib/external/FV3
RUN cat /fv3gfs-wrapper/lib/external/FV3/makefile
RUN PREFIX=/usr/local make -C /fv3gfs-wrapper/lib/external/FV3 install

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
FROM fv3gfs-wrapper-build AS fv3gfs-fv3net

# copy and install dependency packages
COPY /external/fv3config /fv3gfs-wrapper/external/fv3config
COPY /external/fv3gfs-util /fv3gfs-wrapper/external/fv3gfs-util

COPY /external/fv3gfs-wrapper/requirements_local.txt /fv3gfs-wrapper/requirements_local.txt
RUN cd /fv3gfs-wrapper && pip3 install --no-deps --no-cache-dir -r /fv3gfs-wrapper/requirements_local.txt

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

FROM fv3gfs-fv3net AS test

RUN cd /fv3gfs-wrapper && \
    make test
