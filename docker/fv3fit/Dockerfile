FROM python:3.7.8-stretch

COPY external/fv3fit/requirements.txt fv3fit/requirements.txt
RUN pip install -r fv3fit/requirements.txt


COPY external/vcm/ external/vcm/
COPY external/loaders/ external/loaders/
COPY external/synth external/synth
RUN pip install --no-deps external/vcm external/loaders external/synth

COPY external/fv3fit/ fv3fit/
RUN pip install --no-deps -e fv3fit

RUN pytest \
        --cov-config=.coveragerc \
        --cov=fv3fit \
        --cov-report term \
        --cov-report html:htmlcov \
        fv3fit/tests
