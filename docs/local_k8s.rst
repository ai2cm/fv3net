.. _local_k8s:

Local development of argo workflows
===================================

The workflows in this repository can be developed locally. Local development
can have a faster iteration cycle because pushing/pulling images from GCR is
no longer necessary. Also, it avoids cluttering the shared kubernetes cluster
with development resources. Local development works best on a Linux VM with
at least 4 cpus and >10 GB of RAM.

Local development has a few dependencies:

#. a local kubernetes insallation. (see docs below)
#. `kustomize <https://kubernetes-sigs.github.io/kustomize/installation/binaries/>`_ (>= v3.54). The version bundled with kubectl is not recent enough.
#. argo (>= v2.11.6)

Local development requires a local installation of kubernetes. On an ubuntu
system, `microk8s <https://microk8s.io/>`_ is recommended because it is easy to
install on a Google Cloud Platform VM. To install on an ubuntu system run

..  code-block:: bash

    # basic installation
    sudo snap install microk8s --classic

    microk8s status --wait-ready

    # needed plugins
    microk8s enable dashboard dns registry:size=40Gi

To configure kubectl and argo use this local cluster, you need to add microk8s
configurations to the global kubeconfig file. Running `microk8s config` will
display the contents of this file. It is simplest to overwrite any existing k8s
configurations by running:

..  code-block:: bash

    microk8s config > ~/.kube/config

If however, you want to submit jobs to both microk8s and the shared cluster,
you can manually merge the `clusters`, `contexts`, and `users` sections printed by
microk8s config into the the global `~/.kube/config`. Once finished, the file
should something like this (except for the certificate, tokens, and IP
addresses).

..  code-block:: JSON

    apiVersion: v1
    clusters:
    - cluster:
        certificate-authority: ~/proxy.crt
        server: https://12.34.45.67.240
    name: gke_vcm-ml_us-central1-c_ml-cluster-dev
    - cluster:
        certificate-authority-data: SECRETXXXXXXXXXXXXXXXXXXXXX
        server: https://10.128.0.2:16443
    name: microk8s-cluster
    contexts:
    - context:
        cluster: gke_vcm-ml_us-central1-c_ml-cluster-dev
        user: gke_vcm-ml_us-central1-c_ml-cluster-dev
    name: gke_vcm-ml_us-central1-c_ml-cluster-dev
    - context:
        cluster: microk8s-cluster
        user: admin
    name: microk8s
    current-context: microk8s
    kind: Config
    preferences: {}
    users:
    - name: admin
    user:
        token: SECRETXXXXXXX
    - name: gke_vcm-ml_us-central1-c_ml-cluster-dev
    user:
        auth-provider:
        config:
            access-token: SECRETXXXXX
            cmd-args: config config-helper --format=json
            cmd-path: /snap/google-cloud-sdk/156/bin/gcloud
            expiry: "2020-10-28T00:49:43Z"
            expiry-key: '{.credential.token_expiry}'
            token-key: '{.credential.access_token}'
        name: gcp


Then you can switch between contexts using

..  code-block:: bash

    # switch to local microk8s
    kubectl config use-context microk8s

    # switch back the shared cluster
    kubectl config use-context gke_vcm-ml_us-central1-c_ml-cluster-dev

At this point, you should have a running microk8s cluster and your kubectl
configure to refer to it. You can check this be running ``kubectl get node`` and
see if this printout is the same as it was on the GKE cluster. If succesful,
the commands above will start a docker registry process inside of the cluster
than can be used by kubernetes pods. By default the network address for this
registry is ``localhost:32000``. To build and push all the docker images to
this local repository run::

    REGISTRY=localhost:32000 make push_images

To install argo in the cluster and other necessary resources, you first need
to have a GCP service account key file pointed to by the
`GOOGLE_APPLICATION_CREDENTIALS` environmental variable (see 
:ref:`instructions <cloud_auth>`).::

    make deploy_local

Finally, to run the integration tests (which also deploys argo and all the
necessary manifests), you can run::

    REGISTRY=localhost:32000 make run_integration_tests