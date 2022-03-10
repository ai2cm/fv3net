ARG BEAM_VERSION
FROM apache/beam_python3.8_sdk:${BEAM_VERSION}


COPY constraints.txt /etc/
ENV PIP_CONSTRAINT=/etc/constraints.txt

RUN apt-get update && apt-get install -y gfortran
COPY external/vcm /tmp/vcm
RUN pip install -e /tmp/vcm

COPY workflows/dataflow /tmp/dataflow
RUN pip install -e /tmp/dataflow

COPY workflows/fine_res_budget /tmp/fine_res_budget
RUN pip install -e /tmp/fine_res_budget

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=${COMMIT_SHA_ARG}