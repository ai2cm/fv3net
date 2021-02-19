.. _quickstarts:

Quickstarts
===========

Here are some quick setup items for running and developing with ``fv3net``.

Installation
------------

To install all the requirements and fv3net packages locally, run this from
the main directory::

    make create_env

This will incorporate a deterministic build based on frozen pip and conda
requirements files.  See <Reference Dev Environment for more details>


Cloud Authentication
--------------------

The `fv3net` project currently utilizes Google Cloud to deploy workflow items
to services such as Kubernetes and Dataflow.  Authentication requires an
installation of [Google Cloud SDK](https://cloud.google.com/sdk/docs/install).

#. Step 1. Example:

    .. code-block:: bash

      Example code

#. Step 2.

#. Activate your account or service account 
   ([more details](https://cloud.google.com/sdk/docs/authorizing)):
   #. Local Machine::

        > gcloud init
        > gclout auth login
  
   #. Virtual Machine::

        > gcloud auth activate-service-account <account> --key-file=<key-file>

#. Connect to a kubernetes cluster
  a. Pre-existing cluster::

        > gcloud container clusters get-credentials <cluster-name>

  a. From a VM (Vulcan Specific)

    - Clone the 
      [long-lived-infrastructure repo](https://github.com/VulcanClimateModeling/long-lived-infrastructure)
    - Use terraform to connect to our cluster
      [details](https://github.com/VulcanClimateModeling/long-lived-infrastructure#vm-access-setup)::
        
         > make tf_init
         > make tf_dev_workspace_create
         > make kubeconfig_init

    
  b. Local
    i. Proxy firewalled to VM [link](https://github.com/VulcanClimateModeling/long-lived-infrastructure#cluster-provisioning-resettingupdating-kubernetes-cluster)
    ii. local k8s cluster <link to local k8s install

After authenticated you will be able to set up / utilize infrastructure with
proper permissions.  If you are having trouble with authentication being 
recognized check out the `faqs`_ page.

Test drive
----------

Many of the ML workflows are documented with their usage.  Here's a simple example to validate
everything is working properly!
