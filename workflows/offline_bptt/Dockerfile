FROM python:3.7.8-stretch AS bld

COPY constraints.txt /external/constraints.txt
COPY workflows/offline_bptt/requirements.txt /workflows/offline_bptt/requirements.txt
RUN pip install -c /external/constraints.txt -r /workflows/offline_bptt/requirements.txt

COPY external/vcm/ /external/vcm/
COPY external/loaders/ /external/loaders/
COPY external/synth /external/synth
COPY external/fv3fit/ /fv3fit/

COPY docker/fv3fit/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh && \
    /entrypoint.sh

# we will bind-mount this directory when we use the dev environment
# by not copying now we avoid copying data/output files
WORKDIR /workflows/offline_bptt
ENTRYPOINT [ "/entrypoint.sh" ]
