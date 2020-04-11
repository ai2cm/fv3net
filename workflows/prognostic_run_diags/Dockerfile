from ubuntu:18.04

LABEL maintainer="noahb@vulcan.com"

RUN apt-get update && apt-get install -y python3-pip python3

ENV PATH=/opt/conda/bin/:${PATH}
RUN pip3 install toolz xarray netcdf4

# copy in script
COPY metrics.py /metrics.py
RUN chmod +x /metrics.py

# public interface
ENTRYPOINT ["python3", "/metrics.py"]
