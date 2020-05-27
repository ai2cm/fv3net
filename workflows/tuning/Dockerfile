FROM us.gcr.io/vcm-ml/fv3net


# install mongodb and python requirements
USER root
RUN apt update && apt install -y mongodb
USER $NB_USER
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
