FROM continuumio/miniconda3:23.3.1-0

RUN conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba

RUN conda install -c conda-forge -y \
    xarray gcsfs zarr netCDF4 setuptools wheel

RUN conda install -c conda-forge -y eccodes cdo python-cdo && conda clean -tip

RUN pip install 'apache-beam[gcp]' xarray-beam

# Copy files from official SDK image, including script/dependencies.
COPY --from=apache/beam_python3.10_sdk:2.48.0 /opt/apache/beam /opt/apache/beam

# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]
