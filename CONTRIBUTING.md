# Contributions Guides and Standards


## fv3net

- Only imports to `vcm`, `vcm.cubedsphere`, and `vcm.cloud` are allowed. No
  deeper nesting (e.g. `vcm.cloud.fsspec`) or imports to other modules are
  allowed.


##  vcm

- The external interfaces are the modules `vcm`, `vcm.cubedsphere`, and
  `vcm.cloud`. All routines to be used externally should be imported into one
   of these namespaces. This rule could change pending future changes to the vcm API.


