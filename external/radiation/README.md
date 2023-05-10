# Radiation Driver

A python port of the GFS physics radiation scheme. 

See sphinx [documentation](https://vulcanclimatemodeling.com/docs/radiation) for full usage.

### Running validation tests

The following test verifies that the port produces the same output as the GFS scheme
in Fortran.

- Install nix: https://nixos.org/download.html.

- Enter a nix-shell:

    nix-shell

- Download the data via:
```
./get_data.sh
```

- Running radiation driver test:
```
python tests/test_driver.py
```

Driver [input](https://github.com/ai2cm/fv3gfs-fortran/blob/5bec365a6de0f5255e11aaf9dd599f901bba9b92/FV3/gfsphysics/GFS_layer/GFS_radiation_driver.F90#L1277) and [output](https://github.com/ai2cm/fv3gfs-fortran/blob/5bec365a6de0f5255e11aaf9dd599f901bba9b92/FV3/gfsphysics/GFS_layer/GFS_radiation_driver.F90#L2355) data are serialized from running fv3gfs-fortran. Once the `./get_data.sh` script has
been called once, an input dataset closer to the typical fv3net configuration can be
downloaded with:
```
export USE_DIFFERENT_TEST_CASE=true
./get_data.sh
```
and then the tests run as before. 