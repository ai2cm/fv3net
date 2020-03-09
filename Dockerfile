FROM jupyter/base-notebook


ENV ENVIRONMENT_SCRIPTS=$FV3NET/.environment-scripts
ENV PROJECT_NAME=fv3net

# Install dependencies (slow)
USER root
RUN apt-get update && apt-get install -y gfortran
ENV FV3NET=/home/$NB_USER/fv3net
ADD environment.yml  $FV3NET/
ADD Makefile  $FV3NET/
ADD .environment-scripts $ENVIRONMENT_SCRIPTS
RUN fix-permissions $FV3NET
WORKDIR $FV3NET

USER $NB_UID

ENV PATH=/opt/conda/envs/fv3net/bin:$PATH
RUN bash $ENVIRONMENT_SCRIPTS/build_environment.sh $PROJECT_NAME
RUN bash $ENVIRONMENT_SCRIPTS/install_local_packages.sh $PROJECT_NAME

# Add rest of fv3net directory
USER root 
ADD . $FV3NET
RUN fix-permissions $FV3NET
USER $NB_UID

# setup the local python packages
RUN make install_local_packages

