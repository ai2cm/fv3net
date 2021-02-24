FROM jupyter/base-notebook as build

# Manually increment this string to invalidate the cache
ENV MANUAL_CACHE_STRING 1

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
COPY constraints.txt pip-requirements.txt conda-linux-64.lock ./

RUN bash $ENVIRONMENT_SCRIPTS/build_environment.sh $PROJECT_NAME && conda clean -tpi
ENV PATH=/opt/conda/envs/${PROJECT_NAME}/bin:$PATH


# Add rest of fv3net directory
COPY external $FV3NET/external
COPY workflows $FV3NET/workflows

# setup the local python packages
RUN bash $ENVIRONMENT_SCRIPTS/install_local_packages.sh $PROJECT_NAME

# delete any gsutil's installed by conda
RUN rm -f /opt/conda/envs/$PROJECT_NAME/bin/gsutil
RUN fix-permissions /home/jovyan/
USER $NB_UID

# new stage for testing
FROM build AS test
COPY docker/fv3net/test_environment.py /tmp/test_environment.py
RUN pytest -vv /tmp/test_environment.py
RUN touch /tmp/tests-done

# empty stage for final output
FROM build

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG

# add explicit copy to ensure that tests run
COPY --from=test /tmp/tests-done /tmp

# build the docs
COPY docs $FV3NET/docs
USER root
RUN fix-permissions $FV3NET/docs $FV3NET/external/fv3viz/docs
USER $NB_UID
RUN make -C $FV3NET/docs html
RUN make -C $FV3NET/external/fv3viz/docs html
RUN make -C $FV3NET/external/loaders/docs html
RUN make -C $FV3NET/workflows/prognostic_run_diags/docs html
RUN mkdir fv3net_docs
RUN cp -r $FV3NET/docs/_build/html fv3net_docs/fv3net
RUN cp -r $FV3NET/external/fv3viz/docs/_build/html fv3net_docs/fv3viz
RUN cp -r $FV3NET/external/loaders/docs/_build/html fv3net_docs/loaders
RUN cp -r $FV3NET/workflows/prognostic_run_diags/docs/_build/html fv3net_docs/prognostic_run_diags

# add development tools
USER root
RUN apt-get install -y git tar
USER $NB_UID

