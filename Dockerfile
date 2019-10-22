FROM us.gcr.io/vcm-ml/fv3gfs-compiled-default:latest


RUN apt-get update && apt-get install -y \
    python3 \
    python3-netcdf4 \
    cython3

ADD requirements.txt /requirements.txt

RUN pip3 install -r requirements.txt

ADD install_gcloud.sh install_gcloud.sh
RUN bash install_gcloud.sh

