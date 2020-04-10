FROM continuumio/miniconda:4.7.12 as build

COPY environment.yml  /tmp/
RUN conda env create -p /opt/python -f /tmp/environment.yml
ENV PATH=/opt/python/bin:${PATH}
RUN jupyter labextension install @pyviz/jupyterlab_pyviz

FROM google/cloud-sdk:288.0.0

RUN apt-get update && \
    apt-get install -y gfortran jq curl kubectl gettext

# setup docker
COPY --from=build /opt/python /opt/python
ENV PATH=/opt/python/bin:${PATH}

RUN gcloud config set project vcm-ml

# What uses this?
ADD .circleci $FV3NET/.circleci

COPY docker/fv3net/entrypoint.sh /usr/bin/entrypoint.sh
RUN chmod +x /usr/bin/entrypoint.sh
ENTRYPOINT ["/usr/bin/entrypoint.sh"]

# Add rest of fv3net directory
COPY . /fv3net
WORKDIR /fv3net

# setup the local python packages
RUN pip install --no-deps -e . external/fv3config external/vcm external/vcm/external/mappm
