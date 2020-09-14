FROM ubuntu:20.04

ENV DEBIAN_FRONTEND="noninteractive"

RUN  apt-get update && apt-get install -y rsync python3-yaml python3-xarray python3-click python3-toolz python3-fsspec python3-pip
# no package available for gcsfs, installing through pip
RUN pip3 install gcsfs

# install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

RUN apt-get update && apt-get install -y google-cloud-sdk

RUN gcloud config set project vcm-ml

COPY post_process.py /usr/local/bin/
COPY append.py /usr/local/bin/
RUN chmod +x /usr/local/bin/post_process.py
RUN chmod +x /usr/local/bin/append.py

ENTRYPOINT ["post_process.py"]
