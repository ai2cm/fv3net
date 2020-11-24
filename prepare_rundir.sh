write_run_directory /home/andrep/repos/fv3net/scratch/fv3config_full.yaml rundir2

# write a runscript for convenience
cat << EOF > rundir2/run.sh
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/esmf/lib/libO3/Linux.gfortran.64.mpiuni.default/:/FMS/libFMS/.libs/
export TKE_EMU_MODEL=gs://vcm-ml-scratch/andrep/emulation-models/2020-11-09-heat-mom-pbl-classify-ens/keras_model/
mpirun -np 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none fv3.exe > logs 2> err
EOF
