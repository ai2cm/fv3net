python /workdir/rechunk.py /state_out/ /tmp/rechunk/
rm -rf /state_out/state_output.zarr
mv /tmp/rechunk/state_output.zarr /state_out/.