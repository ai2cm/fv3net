FROM jupyter/base-notebook

ENV FV3NET=/home/$NB_USER/fv3net

ENV ENVIRONMENT_SCRIPTS=$FV3NET/.environment-scripts
ENV PROJECT_NAME=fv3net

# Install dependencies (slow)
USER root
RUN apt-get update && \
    apt-get install -y gfortran && \
    apt-get install jq -y
ADD environment.yml  $FV3NET/
ADD .environment-scripts $ENVIRONMENT_SCRIPTS
ADD .circleci $FV3NET/.circleci
RUN fix-permissions $FV3NET
WORKDIR $FV3NET

USER $NB_UID

RUN bash $ENVIRONMENT_SCRIPTS/build_environment.sh $PROJECT_NAME
ENV PATH=/opt/conda/envs/${PROJECT_NAME}/bin:$PATH
RUN jupyter labextension install @pyviz/jupyterlab_pyviz && conda clean -tpi

USER root 
# install gcloud sdk
RUN cd / && \
    curl  https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-284.0.0-linux-x86_64.tar.gz |\
    tar xz
ENV PATH=/google-cloud-sdk/bin:${PATH}
RUN gcloud components install kubectl && \
    gcloud config set project vcm-ml && \
    gcloud auth list && exit 1 
    # gcloud container clusters get-credentials \
    #     --zone $(gcloud container clusters list --format="value(location)") \
    #     --internal-ip \
    #     $(gcloud container clusters list --format="value(name)") && \
    # fix-permissions /home/$NB_USER/.kube

# Add rest of fv3net directory
ADD . $FV3NET

RUN fix-permissions $FV3NET
USER $NB_UID

# setup the local python packages
RUN bash $ENVIRONMENT_SCRIPTS/install_local_packages.sh $PROJECT_NAME
