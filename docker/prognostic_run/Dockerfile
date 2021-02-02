# syntax=docker/dockerfile:experimental
FROM us.gcr.io/vcm-ml/fms-build@sha256:6ac3fee7bbea8213804abd70c15d100b2b81c81c8247990b368d1047aa40289b AS bld

ENV FMS_DIR=/FMS
ENV ESMF_DIR=/usr/local/esmf
ENV FV3GFS_FORTRAN_DIR=/external/fv3gfs-fortran
ENV ESMF_INC="-I${ESMF_DIR}/include -I${ESMF_DIR}/mod/modO3/Linux.gfortran.64.mpiuni.default/"

ENV FMS_LIB=${FMS_DIR}/libFMS/.libs/
ENV ESMF_LIB=${ESMF_DIR}/lib/libO3/Linux.gfortran.64.mpiuni.default/
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_LIB}:${FMS_LIB}

COPY --from=us.gcr.io/vcm-ml/esmf-build@sha256:981815d85ca19999a585fe3e6430c8cd4fd9efd898ad7b9fa3b391e5ede63cf5 /esmf ${ESMF_DIR}
COPY --from=us.gcr.io/vcm-ml/esmf-build@sha256:981815d85ca19999a585fe3e6430c8cd4fd9efd898ad7b9fa3b391e5ede63cf5 /esmf/lib/libO3/Linux.gfortran.64.mpiuni.default/*.so* ${ESMF_LIB}/

# build/install the fortran model
COPY external/fv3gfs-fortran/ /tmp/fortran-build
RUN cd /tmp/fortran-build/FV3 && \
    make -j 8 && \
    PREFIX=/usr/local make install

#
# Python Stuff Here
#

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && \
    apt-get install -y  --no-install-recommends \
    python3 \
    libpython3-dev \
    python3-dev \
    python3-setuptools \
    python3-pip \
    cython3

# Python components here
COPY docker/prognostic_run/requirements.txt /tmp/requirements.txt
COPY constraints.txt /tmp/constraints.txt

RUN pip3 install wheel && \
    pip3 install --upgrade pip==20.2.4 && \
    pip3 install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements.txt && \
    ln -s /bin/python3 /bin/python && \
    ln -s /bin/pip3 /bin/pip

# Build the wrapper
RUN apt-get install -y pkg-config
COPY /external/fv3gfs-wrapper /fv3gfs-wrapper
RUN make -C /fv3gfs-wrapper/lib

# copy dependency packages
COPY /external/fv3gfs-util /external/fv3gfs-util
RUN pip3 install /external/fv3gfs-util 


# cache external package installation
RUN mkdir -p /fv3net/external /fv3net/workflows && \
    mv /external/* /fv3net/external/ && \
    mv /fv3gfs-wrapper /fv3net/external/fv3gfs-wrapper
ENV MPI=mpich
RUN pip3 install /fv3net/external/fv3gfs-wrapper

COPY external/vcm /fv3net/external/vcm
RUN pip3 install /fv3net/external/vcm

COPY external/loaders /fv3net/external/loaders
RUN pip3 install /fv3net/external/loaders

COPY external/fv3fit /fv3net/external/fv3fit
RUN pip3 install /fv3net/external/fv3fit

COPY external/fv3kube /fv3net/external/fv3kube
RUN pip3 install /fv3net/external/fv3kube

COPY workflows/prognostic_c48_run/ /fv3net/workflows/prognostic_c48_run

RUN echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

# Install fv3gfs run script
RUN apt-get install -y jq
RUN pip3 install yq==2.11.1
COPY workflows/prognostic_c48_run/runfv3.sh /usr/local/bin/
RUN chmod 755 /usr/local/bin/runfv3.sh

#Install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk jq python3-dev python3-pip kubectl gfortran


# Install post_process
COPY workflows/post_process_run /fv3net/workflows/post_process_run
RUN pip3 install -c /tmp/constraints.txt -e /fv3net/workflows/post_process_run

# these are needed for "click" to work
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH=/fv3net/workflows/prognostic_c48_run:${PYTHONPATH}
WORKDIR /fv3net/workflows/prognostic_c48_run
CMD ["bash"]

FROM bld as test

ENV GOOGLE_APPLICATION_CREDENTIALS /tmp/key.json
RUN --mount=type=secret,id=gcp,dst=/tmp/key.json \
    ["bash", "-o", "pipefail", "-c", "pytest | tee /tmp/test-results.txt"]

FROM bld
# Copy results from test stage to ensure docker buildkit executes it
COPY --from=test /tmp/test-results.txt .

# build the docs
RUN make -C docs html

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG
