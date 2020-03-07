FROM jupyter/base-notebook

USER root
RUN apt-get update && apt-get install -y gfortran
ENV FV3NET=/home/$NB_USER/fv3net
ADD . $FV3NET
RUN fix-permissions $FV3NET

USER $NB_UID

RUN conda install -y make
RUN make -C $FV3NET create_environment

ENV PATH=/opt/conda/envs/fv3net/bin:$PATH