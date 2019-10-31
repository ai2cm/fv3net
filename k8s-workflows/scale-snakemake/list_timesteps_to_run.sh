files=$(gsutil ls gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/coarsened/C48)
timesteps=$(for f in $files; do basename $f; done )
echo $timesteps
