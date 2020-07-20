FROM jupyter/base-notebook as build


# Install dependencies (slow)
USER root

# install gcloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -


RUN apt-get update && apt-get install -y google-cloud-sdk jq python3-dev python3-pip kubectl gfortran

# install Argo CLI
RUN curl -sSL -o /usr/local/bin/argo https://github.com/argoproj/argo/releases/download/v2.7.0/argo-linux-amd64
RUN chmod +x /usr/local/bin/argo

# install gcloud sdk
RUN gcloud config set project vcm-ml

ENV FV3NET=/home/$NB_USER/fv3net
ENV ENVIRONMENT_SCRIPTS=$FV3NET/environment-scripts
ENV PROJECT_NAME=fv3net

WORKDIR $FV3NET

ADD .environment-scripts $ENVIRONMENT_SCRIPTS
ADD environment.yml  environment.yml

RUN bash $ENVIRONMENT_SCRIPTS/build_environment.sh $PROJECT_NAME && conda clean -tpi
ENV PATH=/opt/conda/envs/${PROJECT_NAME}/bin:$PATH


# Add rest of fv3net directory
COPY external $FV3NET/external
COPY fv3net $FV3NET/fv3net
COPY workflows $FV3NET/workflows
COPY catalog.yml dataflow.sh $FV3NET/

# setup the local python packages
RUN bash $ENVIRONMENT_SCRIPTS/install_local_packages.sh $PROJECT_NAME

# delete any gsutil's installed by conda
RUN rm -f /opt/conda/envs/$PROJECT_NAME/bin/gsutil
RUN fix-permissions $FV3NET
USER $NB_UID
ENTRYPOINT [ "/bin/bash"]

# new stage for testing
FROM build AS test
COPY docker/fv3net/test_environment.py /tmp/test_environment.py
RUN python3 /tmp/test_environment.py

# empty stage for final output
FROM build
