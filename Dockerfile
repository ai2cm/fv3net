FROM jupyter/base-notebook

USER root
RUN apt-get update && apt-get install -y gfortran
ENV UWNET=/home/$NB_USER/uwnet
ADD . $UWNET
RUN fix-permissions $UWNET

USER $NB_UID

RUN conda install -y make
RUN make -C $UWNET create_environment

ENV PATH=/opt/conda/envs/fv3net/bin:$PATH