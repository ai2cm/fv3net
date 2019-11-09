

if [ ! -d 2019-10-05-coarse-grids-and-orography-data/ ] 
then
	gsutil cp gs://vcm-ml-data/2019-10-05-coarse-grid-and-orography-data.tar grid.tar
	tar xf grid.tar
	rm grid.tar
fi




bindMounts="-v /home/noahb:/home/noahb -w $(pwd) -ti "

mosaic=$(pwd)/2019-10-05-coarse-grids-and-orography-data/C48/grid_spec.nc
ls $mosaic


docker run $bindMounts us.gcr.io/vcm-ml/fretools  bash


# fregrid \
#  --input_mosaic $mosaic \
#  --nlat 180 --nlon 90 \
#  --remap_file c34_to_180x90.nc \
#  --input_file  rundir/grid_spec \
#  --output_file area.nc \
#  --scalar_field areat
