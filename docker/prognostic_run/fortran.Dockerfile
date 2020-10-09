FROM ubuntu:20.04 AS fv3gfs-environment

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


## Pull FMS and ESMF
##---------------------------------------------------------------------------------
# these images are defined in fv3gfs-fortran
# KEEP UPDATED IN BOTH DOCKERFILES
FROM us.gcr.io/vcm-ml/fms-build@sha256:868e79a8ef4921f655a6f1fdd32c4eaa200d1157b076a4f85587689bb892e64c AS fv3gfs-fms
FROM us.gcr.io/vcm-ml/esmf-build@sha256:d3154afc4f4b57f9c253be1d84c405b4a3ac508eebbfe5cd0a8c91f65a8287be AS fv3gfs-esmf


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
