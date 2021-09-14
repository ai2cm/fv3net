FROM ubuntu:20.04

RUN apt-get update && apt-get install -y python3 python3-pip
COPY constraints.txt constraints.txt
ENV PIP_CONSTRAINT=/constraints.txt

COPY external/artifacts /tmp/artifacts
RUN pip3 install /tmp/artifacts