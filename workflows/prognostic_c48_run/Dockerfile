FROM us.gcr.io/vcm-ml/fv3gfs-python:v0.2.1

ADD fv3net-0.1.0-py3-none-any.whl /wheels/fv3net-0.1.0-py3-none-any.whl
ADD vcm-0.1.0-py3-none-any.whl /wheels/vcm-0.1.0-py3-none-any.whl
RUN pip3 install --no-deps /wheels/fv3net-0.1.0-py3-none-any.whl && \
    pip3 install /wheels/vcm-0.1.0-py3-none-any.whl && \
    pip3 install scikit-learn==0.22.1 joblib zarr

COPY online_modules/ /online_modules
ENV PYTHONPATH=/online_modules:${PYTHON_PATH}
