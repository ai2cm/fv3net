Release Instructions
====================

## Release naming guidelines

If `0.x.y` is the version, we bump `x` on either big new features or breaking changes, and `y` on smaller changes.

## Release steps

1. Prepare master branch for release (make sure all PRs are merged and tests pass).

2. Run `bumpversion <major/minor/bugfix>`. This will create a new tagged commit,
   having updated all version references to the new, higher version.

## If pushes to master are allowed

3. Run `git push && git push origin --tags` to push the version reference commit and
   then all local tags to master.

## If pushes to master are *not* allowed

3. Create a branch for the version bump and push to Github to open a PR. 
4. Merge this branch to master on Github and create a release version consistent 
   with the version bump.
