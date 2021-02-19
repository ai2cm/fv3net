.. fv3net documentation master file, created by
   sphinx-quickstart on Tue Mar 31 22:39:27 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fv3net's documentation!
==================================

Other VCM Projects
------------------

- fv3gfs-python_ (`docs <https://vulcanclimatemodeling.github.io/fv3gfs-python/f12n7eq5xkoibbqp/index.html>`_)
- fv3config_


short, quick hits of information with links to dive into the full docs
links to paper docs with active descriptions of stored datasets, active experiments
   in that paper doc, please link to the configuration used to generate the dataset

Link to Quickstarts page
Quickstarts:
- setup
   - local environment install
   - building docker containers
   - authentication
      - for each step, ideally include what to do, how to validate that it worked, and how to roll back (rollback could mean delete and start over)
- examples
   - link to vcm-workflow-control
   - running workflows
      - the fact that you probably look in the makefile and README
      - make branch in vcm-workflow-control and edit example folder, so you can easiliy diff with example

Workflows:
- overall description/data flow between workflows
- links to individual workflow docs

Packages:
- for each one, a short description and a link

FAQs:
   - I see this error, what does it mean? (especially for authentication)
   - some kind of "if you get this authentication error, see the authentication rollback and quickstart docs (link)"



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   workflows
   vcm
   science
   regression-testing


.. _fv3gfs-python: https://github.com/VulcanClimateModeling/fv3gfs-python
.. _fv3config: https://fv3config.readthedocs.io/en/latest/

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
