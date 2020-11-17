FROM ubuntu:19.10

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    git \
    libblas-dev \
    liblapack-dev \
    libnetcdf-dev \
    netcdf-bin \
    libnetcdff-dev \
    libmpich-dev \
    autoconf \
    perl \
    make \
    rsync \
    python3-dev \
    python3-pip

# install FRE-NCtools including fregrid
COPY external/FRE-NCtools /FRE-NCtools
RUN cd FRE-NCtools && export CFLAGS=-Wno-traditional && \autoreconf -i && \
    ./configure && make && make install

# install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

RUN apt-get update && apt-get install -y google-cloud-sdk
RUN gcloud config set project vcm-ml

# install fv3post dependencies
RUN pip3 install pytest
COPY external/fv3post/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# install fv3post
COPY external/fv3post /fv3post
RUN cd /fv3post && pip3 install . --no-deps --no-cache-dir

# run fv3post tests
RUN cd /fv3post && pytest && rm -rf /fv3post

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG