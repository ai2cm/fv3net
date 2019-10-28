FROM us.gcr.io/vcm-ml/fv3gfs-compiled-default:latest


RUN apt-get update && apt-get install -y \
    python3 \
    python3-netcdf4 \
    cython3


ADD install_gcloud.sh install_gcloud.sh
RUN bash install_gcloud.sh

ADD download_inputdata.sh download_inputdata.sh
RUN bash download_inputdata.sh

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /code
ENV PYTHONPATH=/code:$PYTHONPATH
WORKDIR /code

