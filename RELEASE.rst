====================
Release Instructions
====================

Release naming guidelines
-------------------------

If `x.y.z` is the version, bump `y` (minor) on new features or breaking changes, and `z` on smaller changes.

Release steps
-------------

1. Prepare master branch for release (make sure all PRs are merged and tests pass).

2. Create a branch for the version bump and run `bumpversion <part>` where part
   is the section of the version to bump. I.e., for x.y.z, the corresponding "part"
   would be "major" for x, "minor" for y, and "patch" for z.
   This will create a new commit, having updated all version references to the new, 
   higher version.

3. Push the bump branch to Github to open a PR. 

3. Merge this branch to master on Github and create a release version consistent 
   with the version bump.
