FROM 3.8.3-buster

pip install --no-cache-dir pyyaml

# install mongodb and python requirements
USER root
RUN apt update && apt install -y mongodb
USER $NB_USER
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
