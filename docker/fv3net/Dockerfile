FROM jupyter/base-notebook


# Install dependencies (slow)
USER root

# install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -


RUN apt-get update && apt-get install -y google-cloud-sdk jq python3-dev python3-pip kubectl gfortran

# install gcloud sdk
RUN gcloud config set project vcm-ml

ENV FV3NET=/home/$NB_USER/fv3net
ENV ENVIRONMENT_SCRIPTS=$FV3NET/environment-scripts
ENV PROJECT_NAME=fv3net

WORKDIR $FV3NET
USER $NB_UID

ADD .environment-scripts $ENVIRONMENT_SCRIPTS
ADD environment.yml  environment.yml

RUN bash $ENVIRONMENT_SCRIPTS/build_environment.sh $PROJECT_NAME && conda clean -tpi
ENV PATH=/opt/conda/envs/${PROJECT_NAME}/bin:$PATH

# delete any gsutil's installed by conda
RUN rm -f /opt/conda/envs/$PROJECT_NAME/gsutil

# Add rest of fv3net directory
COPY . $FV3NET
USER root
RUN fix-permissions $FV3NET
RUN mkdir -p /home/jovyan/.config
RUN fix-permissions /home/jovyan/.config
USER $NB_UID

# setup the local python packages
RUN bash $ENVIRONMENT_SCRIPTS/install_local_packages.sh $PROJECT_NAME
RUN pip install apache_beam[gcp] six==1.12.0
CMD ["/bin/bash"]
