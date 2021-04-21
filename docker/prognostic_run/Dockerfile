# syntax=docker/dockerfile:experimental
FROM nixos/nix as bld

RUN nix-env -iA cachix -f https://cachix.org/api/v1/install
RUN cachix use vulcanclimatemodeling
RUN nix-env -iA nixpkgs.git nixpkgs.openssh
COPY shell.nix /tmp/shell.nix
SHELL ["nix-shell", "/tmp/shell.nix", "--run"]

# run the nix-shell once for caching
RUN echo "caching"

FROM bld as next

# Python components here
COPY docker/prognostic_run/requirements/base.txt /tmp/requirements.txt
COPY constraints.txt /tmp/constraints.txt

# Enforce constraints are used for all pip installs
ENV PIP_CONSTRAINT=/tmp/constraints.txt
# Equivalent to passing --no-cache-dir to every pip install
ENV PIP_NO_CACHE_DIR=off

# Build the wrapper
COPY /external/fv3gfs-wrapper /fv3gfs-wrapper
RUN make -C /fv3gfs-wrapper/lib

# copy dependency packages
COPY /external/fv3gfs-util /external/fv3gfs-util
RUN pip install /external/fv3gfs-util 

# cache external package installation
RUN mkdir -p /fv3net/external /fv3net/workflows && \
    mv /external/* /fv3net/external/ && \
    mv /fv3gfs-wrapper /fv3net/external/fv3gfs-wrapper
ENV MPI=mpich
RUN pip install /fv3net/external/fv3gfs-wrapper

COPY external/vcm /fv3net/external/vcm
RUN pip install /fv3net/external/vcm

COPY external/loaders /fv3net/external/loaders
RUN pip install /fv3net/external/loaders

COPY external/fv3fit /fv3net/external/fv3fit
RUN pip install /fv3net/external/fv3fit

COPY external/fv3kube /fv3net/external/fv3kube
RUN pip install /fv3net/external/fv3kube

RUN echo "ulimit -s unlimited" >> /etc/bash.bashrc && \
    mkdir /outdir && \
    chmod -R 777 /outdir

# Install fv3gfs run script
RUN pip install yq==2.11.1
COPY workflows/prognostic_c48_run/runfv3 /usr/local/bin/
RUN chmod 755 /usr/local/bin/runfv3

# Install post_process
COPY workflows/post_process_run /fv3net/workflows/post_process_run
RUN pip install -c /tmp/constraints.txt -e /fv3net/workflows/post_process_run

# these are needed for "click" to work
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY workflows/prognostic_c48_run/ /fv3net/workflows/prognostic_c48_run
RUN pip install /fv3net/workflows/prognostic_c48_run

ENV PYTHONPATH=/fv3net/workflows/prognostic_c48_run:${PYTHONPATH}
WORKDIR /fv3net/workflows/prognostic_c48_run
CMD ["bash"]

###############################################################################
#  prognostic-run tests
###############################################################################
FROM bld as test

ENV GOOGLE_APPLICATION_CREDENTIALS /tmp/key.json
RUN --mount=type=secret,id=gcp,dst=/tmp/key.json \
    ["bash", "-o", "pipefail", "-c", "pytest | tee /tmp/test-results.txt"]

###############################################################################
#  prognostic-run image
###############################################################################
FROM bld AS prognostic-run
# Copy results from test stage to ensure docker buildkit executes it
COPY --from=test /tmp/test-results.txt .

# build the docs
RUN pip install sphinx sphinx_rtd_theme
RUN make -C docs html

ARG COMMIT_SHA_ARG
ENV COMMIT_SHA=$COMMIT_SHA_ARG

###############################################################################
#  notebook image
###############################################################################
FROM bld AS notebook

COPY docker/prognostic_run/requirements/notebook.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt-get install -y bash-completion vim netcdf-bin

RUN useradd --create-home --shell /bin/bash jovyan

USER jovyan
WORKDIR /home/jovyan
VOLUME /home/jovyan

ENTRYPOINT ["jupyter", "lab"]
CMD ["--ip", "0.0.0.0", "--port", "8888", "--no-browser"]