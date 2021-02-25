# syntax=docker/dockerfile:experimental
FROM python:3.7.8-stretch AS bld

# install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update && apt-get install -y google-cloud-sdk 

COPY constraints.txt /tmp/constraints.txt
COPY external/fv3fit/requirements.txt fv3fit/requirements.txt
RUN pip install -c /tmp/constraints.txt -r fv3fit/requirements.txt

COPY external/vcm/ /external/vcm/
COPY external/loaders/ /external/loaders/
COPY external/synth /external/synth
COPY external/fv3fit/ /fv3fit/

RUN apt-get update && apt-get install -y gfortran

COPY docker/fv3fit/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    /entrypoint.sh

# build the docs
RUN make -C /fv3fit/docs html

WORKDIR /fv3fit
ENTRYPOINT [ "/entrypoint.sh" ]
FROM bld AS test

ENV GOOGLE_APPLICATION_CREDENTIALS /tmp/key.json
RUN --mount=type=secret,id=gcp,dst=/tmp/key.json \
    pytest \
    --cov-config=.coveragerc \
    --cov=fv3fit \
    --cov-report term \
    --cov-report html:htmlcov

RUN apt-get install tar

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG