FROM us.gcr.io/vcm-ml/fv3gfs-wrapper as bld

RUN pip install --upgrade setuptools wheel pip

COPY workflows/train_bptt/requirements.txt /train_bptt/requirements.txt
RUN pip install --no-deps -r train_bptt/requirements.txt

COPY external/vcm/ /external/vcm/
COPY external/loaders/ /external/loaders/
COPY external/synth /external/synth
COPY external/fv3fit/ /external/fv3fit/

COPY workflows/train_bptt/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    /entrypoint.sh

WORKDIR /train_bptt
ENTRYPOINT [ "/entrypoint.sh" ]
