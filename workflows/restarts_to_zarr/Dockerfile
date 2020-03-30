from daskdev/dask

RUN apt-get update && apt-get install -y gcc g++

COPY workflows/restarts_to_zarr/requirements.txt  /
RUN pip install -r /requirements.txt

COPY external/vcm /fv3net/external/vcm
RUN pip install --no-deps -e /fv3net/external/vcm
COPY . /fv3net
RUN pip install --no-deps -e /fv3net
WORKDIR /




