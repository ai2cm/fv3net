FROM python:3.7.8-stretch AS bld

# install gcloud sdk
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update && apt-get install -y google-cloud-sdk
RUN gcloud config set project vcm-ml
ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/key.json

COPY constraints.txt /tmp/constraints.txt
COPY external/fv3fit/requirements.txt fv3fit/requirements.txt
RUN pip install -c /tmp/constraints.txt -r fv3fit/requirements.txt

COPY external/vcm/ /external/vcm/
COPY external/loaders/ /external/loaders/
COPY external/synth /external/synth
COPY external/fv3fit/ /fv3fit/
COPY catalog.yml /fv3fit/catalog.yml

COPY docker/fv3fit/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    /entrypoint.sh

WORKDIR /fv3fit
ENTRYPOINT [ "/entrypoint.sh" ]

FROM bld AS test

RUN pytest \
        --cov-config=.coveragerc \
        --cov=fv3fit \
        --cov-report term \
        --cov-report html:htmlcov \
        /fv3fit/tests

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG