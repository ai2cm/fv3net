FROM us.gcr.io/vcm-ml/fretools

COPY regrid.sh /usr/bin/regrid.sh
RUN chmod +x /usr/bin/regrid.sh

RUN apt-get update && apt-get install -y python3 python3-pip
ADD requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

