.. _quickstarts:

Quickstarts
===========

Here are some quick setup items for running and developing with ``fv3net``.

Installation
------------

To install all the requirements and fv3net packages locally, run this from
the main directory::

    make update_submodules
    make create_environment

This creates an Anaconda environment ``fv3net`` with all the dependicies
for running the Vulcan FV3GFS ML workflows.   All submodules are installed
in development mode (e.g., ``pip install -e``) so any modifications will
be loaded in the conda environment.  After the build completes successfully,
activate the environment with::

    conda activate fv3net

.. _cloud_auth:

Cloud Authentication
--------------------

The `fv3net` project currently utilizes Google Cloud to deploy workflow items
to services such as Kubernetes and Dataflow.  Authentication requires an
installation of `Google Cloud SDK <https://cloud.google.com/sdk/docs/install>`_.

Authentication obtained via ``gcloud auth login`` does not work well with
secrets management and is not used by many APIs. Service account key-based
authentication works much better, because the service account key is a single
file that can be deployed in a variety of contexts (K8s cluter, VM, etc).
Many Python APIs can authenticate with google using the
``GOOGLE_APPLICATION_CREDENTIALS`` environmental variable. 
`(See Google authentication details) <https://cloud.google.com/sdk/docs/authorizing>`_

* If gcloud is a fresh install, initialize and grab a keyfile::
      
    > gcloud init
    > gcloud auth login
    > mkdir -p ~/.keys
    > gcloud iam service-accounts keys create ~/.keys/key.json \
          --iam-account <service account>

* Else activate your service account key::

    > gcloud auth activate-service-account <account> --key-file=<key-file>
    > export GOOGLE_APPLICATION_CREDENTIALS=~/.keys/key.json

It is recommended to add ``GOOGLE_APPLICATION_CREDENTIALS`` to your .bashrc since
many libraries and tools require it.

Connecting to a kubernetes cluster
----------------------------------

  * Pre-existing basic cluster::

      > gcloud container clusters get-credentials <cluster-name>

  * From a VM to our firewalled cluster (Vulcan Specific)

    * Clone the 
      `long-lived-infrastructure repo <https://github.com/VulcanClimateModeling/long-lived-infrastructure>`_
    * Use terraform to connect to our cluster
      `(details) <https://github.com/VulcanClimateModeling/long-lived-infrastructure#vm-access-setup>`_::
        
        > make tf_init
        > make tf_dev_workspace_create
        > make kubeconfig_init
    
  * We also have an detailed explanation of creating and connecting to
    a local k8s cluster in :ref:`local_k8s`. 

After authenticated you will be able to set up / utilize infrastructure with
proper permissions.  If you are having trouble with authentication being 
recognized check out the :ref:`faqs` page.

Test drive
----------

Many of the ML workflows are documented with their usage.  Here's a simple example to validate
everything is working properly!
