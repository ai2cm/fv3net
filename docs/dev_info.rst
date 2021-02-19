.. _dev_info:

Developer information
=====================

Links and extended explanations of tools for development

* :ref:`local_k8s`: Use a local k8s cluster for development
* :ref:`dependency_management`: Adding or updating requirements? Look
  here to help keep our builds deterministic!


Code linting checks
-------------------

This python code in this project is autoformated using the
`black <https://black.readthedocs.io/en/stable/>`_ code formatting tool, and the
`isort <https://github.com/timothycrosley/isort>`_ tool for automatically sorting
the order of import statements. To pass CI, any contributed code must be
unchanged by black and also checked by the flake8 linter. However, please use
isort to sort the import statements (done automatically by `make reformat`
below).

Contributers can see if their *commited* code passes these standards by running::

    make lint

If it does not pass, than it can be autoformatted using::

    make reformat


Building the fv3net docker images
---------------------------------

The workflow depends on docker images of ``fv3net`` tools for cloud deployment. 
These images can be built and pushed to GCR using `make build_images` and
`make push_images`, respectively.


Misc To Deal with Stil
----------------------

# How to contribute to fv3net

Please see the [contribution guide.](./CONTRIBUTING.md)

# How to get updates and releases

For details on what's included or upcoming for a release, please see the [HISTORY.rst](./HISTORY.rst) document.

For instructions on preparing a release, please read [RELEASE.rst](./RELEASE.rst).

