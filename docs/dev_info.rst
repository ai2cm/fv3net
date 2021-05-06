.. _dev_info:

Developer information
=====================

Links and extended explanations of tools for development

.. toctree::
    :maxdepth: 1
    :hidden:

    local_k8s
    dependency_management


* :ref:`local_k8s`: Use a local k8s cluster for development
* :ref:`dependency_management`: Adding or updating requirements? Look
  here to help keep our builds deterministic!

.. _linting:

Code linting checks
-------------------

This python code in this project is autoformated using the `black
<https://black.readthedocs.io/en/stable/>`_ code formatting tool. To pass CI,
any contributed code must be unchanged by black and also checked by the
flake8 linter.

Contributers can see if their *commited* code passes these standards by running::

    make lint

If it does not pass, than it can be autoformatted using::

    make reformat

Pre-commit hooks
~~~~~~~~~~~~~~~~

It is convenient to run these code checks before every git commit. This
repository is configured to use `pre-commit <https://pre-commit.com/>`_ tool
to do this. To install the git hooks, run::

    make setup-hooks

After this, any commits with poorly formatted code will be rejected. For
example, here is what happens when trying to check a large file into git::

    $ dd if=/dev/zero of=empty-data.bin bs=1m count=50
    50+0 records in
    50+0 records out
    52428800 bytes transferred in 0.064860 secs (808339615 bytes/sec)
    $ git add empty-data.bin
    $ git commit -m "trying to check in a massive file"
    Check for added large files..............................................Failed
    - hook id: check-added-large-files
    - exit code: 1

    empty-data.bin (51200 KB) exceeds 250 KB.

    black................................................(no files to check)Skipped
    flake8...............................................(no files to check)Skipped
    flake8 __init__.py files.............................(no files to check)Skipped


.. _docker_images:

Building the fv3net docker images
---------------------------------

The workflow depends on docker images of ``fv3net`` tools for cloud deployment. 
These images can be built and pushed to GCR using `make build_images` and
`make push_images`, respectively.


Unhandled Topics (WIP)
----------------------

* How to contribute to fv3net
    Please see the [contribution guide.](./CONTRIBUTING.md)

* How to get updates and releases
    For details on what's included or upcoming for a release, please see the [HISTORY.rst](./HISTORY.rst) document.

    For instructions on preparing a release, please read [RELEASE.rst](./RELEASE.rst).

