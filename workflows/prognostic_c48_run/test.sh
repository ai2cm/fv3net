# TODO delete this dev script
set -e

python3 ./train.py \
--nfiles 2 \
--epochs=2 \
--levels=25,50,75 --multi-output --relative-humidity --extra-variables=cos_zenith_angle,surface_pressure --rh-weight=10000 --q-weight=0 --lr=0.01 --momentum=0.9  --batch-size=512 --num-hidden=512 --num-hidden-layers=3


python3 ./train.py \
--nfiles 2 \
--epochs=2 \
--levels=25,50,75 --level 50 --variable 3 --extra-variables=cos_zenith_angle,surface_pressure --lr=10 --momentum=0.9  --batch-size=512 --num-hidden=512 --num-hidden-layers=3