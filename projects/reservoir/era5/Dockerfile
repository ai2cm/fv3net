FROM apache/beam_python3.10_sdk:2.51.0

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    git \
    libblas-dev \
    liblapack-dev \
    libnetcdf-dev \
    netcdf-bin \
    libnetcdff-dev \
    libmpich-dev \
    autoconf \
    perl \
    make \
    rsync \
    python3-dev \
    python3-pip

# install FRE-NCtools including fregrid.
RUN git config --global http.sslverify false && \
    git clone https://github.com/NOAA-GFDL/FRE-NCtools.git && \
    cd FRE-NCtools && \
    git checkout 2023.01.01 && \
    export CFLAGS=-Wno-traditional NETCDF_LDFLAGS=-lm && \
    autoreconf -i && ./configure && make && make install && \
    cd .. && rm -r FRE-NCtools

RUN pip install 'apache-beam[gcp]' xarray xarray-beam zarr dask netCDF4 immutabledict rechunker gcsfs h5netcdf

# see previous commit if needing to install CDO for gaussian latlon regridding
RUN mkdir -p /work/source-grid-data/C48 && \
     gsutil -m cp gs://vcm-ml-raw/2020-11-12-gridspec-orography-and-mosaic-data/C48/*.nc /work/source-grid-data/C48 && \
     gsutil cp gs://vcm-ml-scratch/andrep/reservoir/era5/regular_lonlat_*.nc /work/ && \
     chmod -R 777 /work

# Copy files from official SDK image, including script/dependencies.
COPY --from=apache/beam_python3.10_sdk:2.48.0 /opt/apache/beam /opt/apache/beam

# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]
