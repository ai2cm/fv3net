FROM us.gcr.io/vcm-ml/fv3gfs-python@sha256:9403cf78f240c4e38dd98272e1876e044654bf09f8b46abc7dc1a4bd5f2f2ab2

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# cache external package installation
COPY external/fv3config /fv3net/external/fv3config
COPY external/vcm /fv3net/external/vcm
RUN pip3 install -e /fv3net/external/vcm -e /fv3net/external/fv3config

COPY . /fv3net
RUN pip3 install --no-deps -e /fv3net
