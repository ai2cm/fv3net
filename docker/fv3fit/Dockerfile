from python:3.7.8-stretch

COPY external/vcm/ external/vcm/
RUN pip install external/vcm

COPY external/loaders/ external/loaders/
COPY external/synth external/synth
RUN pip install external/loaders external/synth

COPY external/fv3fit/ fv3fit/
RUN pip install -r fv3fit/requirements.txt
RUN pip install -e fv3fit

RUN pytest \
        --cov-config=.coveragerc \
        --cov=fv3fit \
        --cov-report term \
        --cov-report html:htmlcov \
        fv3fit/tests
