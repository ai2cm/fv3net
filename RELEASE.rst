====================
Release Instructions
====================

Release naming guidelines
-------------------------

If `0.x.y` is the version, bump `x` on either big new features or breaking changes, and `y` on smaller changes.

Release steps
-------------

1. Prepare master branch for release (make sure all PRs are merged and tests pass).

2. Create a branch for the version bump and run `bumpversion <major/minor/bugfix>`. 
   This will create a new commit, having updated all version references to the new, 
   higher version.

3. Push the bump branch to Github to open a PR. 

3. Merge this branch to master on Github and create a release version consistent 
   with the version bump.
