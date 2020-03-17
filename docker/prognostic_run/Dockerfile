FROM us.gcr.io/vcm-ml/fv3gfs-python:v0.2.1


COPY docker/prognostic_run/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
COPY fv3net-0.1.0-py3-none-any.whl /wheels/fv3net-0.1.0-py3-none-any.whl
COPY vcm-0.1.0-py3-none-any.whl /wheels/vcm-0.1.0-py3-none-any.whl
RUN pip3 install --no-deps /wheels/fv3net-0.1.0-py3-none-any.whl  && pip3 install /wheels/vcm-0.1.0-py3-none-any.whl
