set -e

if ! $(python3 -c 'import fv3fit')
then
    pip install -e /fv3net/external/vcm
    pip install -e /fv3net/external/fv3fit
fi

rm -fr fv3config1.yml rundir
# name fv3config1.yml to avoid get_hooks error from microphysics emulation
prepare-config base.yaml > fv3config1.yml
runfv3 run-native fv3config1.yml rundir