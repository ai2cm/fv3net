# Contributions Guides and Standards

Please record all changes and updates in [HISTORY.rst](./HISTORY.rst) under the 
upcoming section.  It is especially important to log changes that break backwards 
compatibility so we can appropriately adjust the versioning.

## fv3net

- Only imports to `vcm`, `vcm.cubedsphere`, `vcm.safe`, `vcm.cloud`, `vcm.testing` are allowed. No
  deeper nesting (e.g. `vcm.cloud.fsspec`) or imports to other modules are
  allowed.

### Workflow changes

Excessive coupling between our workflows has several negative consequences including:
- Changes to shared code can potentially break any workflow importing that code, making it hard to predict what can break for a given change.
- Depending on uneeded code can increase the number of dependencies required for a particular workflow. More dependencies result in more bugs related to upstream changes and more setup costs (e.g. huge docker images or slow dataflow startup times).

To make each workflow as modular as possible, we need to make sure that workflow-specific code cannot effect other workflows. This is accomplished by following these rules:

- Any new workflow code is assumed workflow-specific unless actually used by multiple workflows.
- Workflow specific code should be contained within the  `workflows/<workflow name>/` folder.

Of course, some workflow-specific code will become useful in other workflows. In this case, the shared functionality should be moved to a relevant python "micropackage" in `external`. If the new functionality does not seem like a good fit for an existing package e.g. (`external/vcm`) then a new package should be created. This is relatively easy to do with tools like [poetry](https://github.com/python-poetry/poetry). Each of these micropackages should have a minimal set of dependencies. 

##  vcm

- The external interfaces are the modules `vcm`, `vcm.cubedsphere`, and
  `vcm.cloud`. All routines to be used externally should be imported into one
   of these namespaces. This rule could change pending future changes to the vcm API.


