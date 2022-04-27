FROM ubuntu:20.04

RUN apt-get update && apt-get install -y python3 python3-pip curl gzip

# Install argo for cronjob submission
# https://github.com/argoproj/argo-workflows/releases/tag/v2.11.6
RUN curl -sL https://github.com/argoproj/argo-workflows/releases/download/v2.11.6/argo-linux-amd64.gz \
    | gunzip  > /usr/local/bin/argo \
    && chmod +x /usr/local/bin/argo \
    && echo

COPY docker/artifacts/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY external/artifacts /tmp/artifacts
RUN pip3 install --no-cache-dir --no-dependencies /tmp/artifacts

COPY projects/microphysics/argo /projects/microphysics/argo
COPY projects/microphysics/experiments /projects/microphysics/experiments
COPY projects/microphysics/train /projects/microphysics/train
COPY projects/microphysics/configs /projects/microphysics/configs

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG
