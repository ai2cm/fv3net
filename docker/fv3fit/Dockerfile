FROM python:3.7.8-stretch AS bld

COPY constraints.txt /tmp/constraints.txt
COPY external/fv3fit/requirements.txt fv3fit/requirements.txt
RUN pip install -c /tmp/constraints.txt -r fv3fit/requirements.txt

COPY external/vcm/ /external/vcm/
COPY external/loaders/ /external/loaders/
COPY external/synth /external/synth
COPY external/fv3fit/ /fv3fit/

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

FROM bld
