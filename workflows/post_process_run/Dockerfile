FROM ubuntu:20.04

ENV DEBIAN_FRONTEND="noninteractive"

RUN  apt-get update && apt-get install -y python3-yaml python3-xarray python3-click python3-toolz


COPY post_process.py /usr/local/bin/
RUN chmod +x /usr/local/bin/post_process.py

ENTRYPOINT ["post_process.py"]
