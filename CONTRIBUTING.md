# Contributions Guides and Standards

Please record all changes and updates in [HISTORY.rst](./HISTORY.rst) under the 
upcoming section.  It is especially important to log changes that break backwards 
compatibility so we can appropriately adjust the versioning.

## fv3net

- Only imports to `vcm`, `vcm.cubedsphere`, and `vcm.cloud` are allowed. No
  deeper nesting (e.g. `vcm.cloud.fsspec`) or imports to other modules are
  allowed.


##  vcm

- The external interfaces are the modules `vcm`, `vcm.cubedsphere`, and
  `vcm.cloud`. All routines to be used externally should be imported into one
   of these namespaces. This rule could change pending future changes to the vcm API.


