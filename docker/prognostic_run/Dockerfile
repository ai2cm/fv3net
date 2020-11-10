# syntax=docker/dockerfile:experimental
FROM us.gcr.io/vcm-ml/fms-build AS bld

ENV FMS_DIR=/FMS
ENV ESMF_DIR=/usr/local/esmf
ENV FV3GFS_FORTRAN_DIR=/external/fv3gfs-fortran
ENV ESMF_INC="-I${ESMF_DIR}/include -I${ESMF_DIR}/mod/modO3/Linux.gfortran.64.mpiuni.default/"

ENV FMS_LIB=${FMS_DIR}/libFMS/.libs/
ENV ESMF_LIB=${ESMF_DIR}/lib/libO3/Linux.gfortran.64.mpiuni.default/
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}

COPY --from=us.gcr.io/vcm-ml/esmf-build /usr/local/esmf ${ESMF_DIR}
COPY --from=us.gcr.io/vcm-ml/esmf-build /usr/local/esmf/lib/libO3/Linux.gfortran.64.mpiuni.default/*.so* ${ESMF_LIB}/

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && \
    apt-get install -y  --no-install-recommends \
    cmake \
    python3 \
    libpython3-dev \
    python3-dev \
    python3-setuptools \
    python3-pip \
    python3-cffi \
    cython3


# install pfunit
RUN cd /tmp && curl -L https://github.com/Goddard-Fortran-Ecosystem/pFUnit/archive/3.2.9.tar.gz | tar xz
ENV F90=gfortran
ENV F90_VENDOR=GNU
ENV PFUNIT=/usr/local
RUN ls /tmp/pFUnit-3.2.9
RUN cd /tmp/pFUnit-3.2.9 && \
    cmake . && \
    make &&\
    make install INSTALL_DIR=${PFUNIT}

# install call-py-fort
RUN ln -s /usr/bin/python3 /usr/bin/python
COPY external/fv3gfs-fortran/FV3/call_py_fort /opt/call_py_fort
ENV PYTHONPATH=/opt/call_py_fort/src/:/opt/call_py_fort/test:$PYTHONPATH
ENV CALLPY=/opt/call_py_fort
RUN cd /opt/call_py_fort/ && make && make install


# build/install the fortran model
COPY external/fv3gfs-fortran/ /tmp/fortran-build
RUN cd /tmp/fortran-build/FV3 && \
    make -j 8 && \
    PREFIX=/usr/local make install

#
# Python Stuff Here
#

# Python components here
COPY docker/prognostic_run/requirements.txt /tmp/requirements.txt
COPY constraints.txt /tmp/constraints.txt

RUN pip3 install wheel && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements.txt && \
    ln -s /bin/python3 /bin/python && \
    ln -s /bin/pip3 /bin/pip

# Build the wrapper
RUN apt-get install -y pkg-config
COPY /external/fv3gfs-wrapper /fv3gfs-wrapper
RUN make -C /fv3gfs-wrapper/lib

# copy dependency packages
COPY /external/fv3config /external/fv3config
COPY /external/fv3gfs-util /external/fv3gfs-util

RUN echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

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

# Install fv3gfs run script
COPY workflows/prognostic_c48_run/runfv3.sh /usr/local/bin/
RUN chmod 755 /usr/local/bin/runfv3.sh

# install the packages
RUN /entrypoint.sh
CMD ["bash"]

ENV PYTHONPATH=/fv3net/workflows/prognostic_c48_run:${PYTHONPATH}
WORKDIR /fv3net/workflows/prognostic_c48_run
ENTRYPOINT [ "/entrypoint.sh" ]

FROM bld as test

ENV GOOGLE_APPLICATION_CREDENTIALS /tmp/key.json
RUN --mount=type=secret,id=gcp,dst=/tmp/key.json \
    ["bash", "-o", "pipefail", "-c", "pytest | tee /tmp/test-results.txt"]

FROM bld
# Copy results from test stage to ensure docker buildkit executes it
COPY --from=test /tmp/test-results.txt .

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG