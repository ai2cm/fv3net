FROM fv3fit:latest as bld

RUN pip install -c /tmp/constraints.txt poetry

COPY external/report /poetry/report
COPY workflows/emulation /fv3fit/emulation

RUN cd /poetry/report && \
    poetry build && \
    for wheel in dist/*.whl; do pip install $wheel; done
RUN pip install nc-time-axis>=1.2.0

WORKDIR /fv3fit
ENTRYPOINT [ "/entrypoint.sh" ]

