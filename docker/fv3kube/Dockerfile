FROM python:3.8-slim-buster

COPY constraints.txt constraints.txt
COPY external/fv3config/ fv3config
COPY external/fv3kube fv3kube 

RUN mkdir -p /runfiles
COPY workflows/prognostic_c48_run/sklearn_runfile.py /runfiles/sklearn.py
COPY workflows/prognostic_c48_run/nudge_to_obs/runfile.py /runfiles/nudge_to_obs.py
COPY workflows/prognostic_c48_run/nudging/nudging_runfile.py /runfiles/nudging.py

RUN pip install -c constraints.txt ./fv3config  pytest pytest-regtest && \
    pip install -c constraints.txt ./fv3kube 

RUN cd fv3kube && pytest



