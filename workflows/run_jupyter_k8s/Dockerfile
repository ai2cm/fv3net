FROM us.gcr.io/vcm-ml/fv3net


ADD worker.yaml /home/jovyan/
ADD test.py /home/jovyan/

USER root
RUN fix-permissions /home/jovyan

USER $NB_USER
RUN pip install dask-kubernetes
