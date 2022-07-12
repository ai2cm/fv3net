# Radiation Driver

Install nix: https://nixos.org/download.html.

Enter a nix-shell:

    nix-shell

Download the data via:
```
./get_data.sh
```

Running radiation driver test:
```
# make sure to run inside the python directory
cd python
python test_driver.py
```

Driver [input](https://github.com/ai2cm/fv3gfs-fortran/blob/5bec365a6de0f5255e11aaf9dd599f901bba9b92/FV3/gfsphysics/GFS_layer/GFS_radiation_driver.F90#L1277) and [output](https://github.com/ai2cm/fv3gfs-fortran/blob/5bec365a6de0f5255e11aaf9dd599f901bba9b92/FV3/gfsphysics/GFS_layer/GFS_radiation_driver.F90#L2355) data are serialized from running fv3gfs-fortran. By default, the input data are 24 columns by 63 k-levels, these can be changed in `config.py`.