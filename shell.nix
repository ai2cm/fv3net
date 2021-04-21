let
  pkgs = import (builtins.fetchGit {
    url = "ssh://git@github.com/VulcanClimateModeling/packages";
    ref = "master";
    rev = "932ac417a95a08bf1806696cf921aa6a3e061536";
  });

  my-python = pkgs.python3.withPackages (ps: with ps;[

    # prog run
    fv3gfs-wrapper

    # vcm inputs
    dask
    xgcm
    f90nml
    click
    appdirs
    requests
    h5py
    xarray
    toolz
    scipy
    metpy
    joblib
    intake
    gcsfs
    zarr
    cftime
    pytest
    google-cloud-storage
    google-api-core
    pytest-regtest
    h5netcdf
    docrep
    # intake-xarray

  ]);
  
in 
with pkgs;
mkShell {
  venvDir = "./.venv";
  buildInputs = [
    python3Packages.venvShellHook
    jq
    google-cloud-sdk
    graphviz
    my-python

    # this is key: https://discourse.nixos.org/t/building-shared-libraries-with-fortran/11876/2
    gfortran.cc.lib
    gfortran
  ];

  postShellHook = ''
    export MPI=mpich
    export PYTHONPATH=$(pwd)/external/vcm:$PYTHONPATH
  '';
}
