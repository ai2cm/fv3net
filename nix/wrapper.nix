{buildPythonPackage, fv3, cython, jinja2, pkgconfig, numpy, pkg-config, gfortran
  , fms
  , esmf
  , nceplibs
  , netcdf
  , netcdffortran
  , lapack
  , blas
  , mpi4py
  , pyyaml
  , fv3gfs-util
  , netcdf4
}:
buildPythonPackage rec {
      version = "0.1.0";
      pname = "fv3gfs-wrapper";

      src = ../external/fv3gfs-wrapper;
      checkInputs = [];

      # environmental variables needed for the wrapper
      PKG_CONFIG_PATH="${fv3}/lib/pkgconfig";
      MPI = "mpich";

      buildInputs = [
          fms
          esmf
          nceplibs
          netcdf
          netcdffortran
          lapack
          blas
          mpi4py.mpi
          # this is key: https://discourse.nixos.org/t/building-shared-libraries-with-fortran/11876/2
          gfortran.cc.lib
          gfortran
      ];

      nativeBuildInputs =  [
          pkg-config
          pkgconfig
          jinja2
          cython
          fv3
          gfortran
      ];


      preBuild = ''
      echo "RUNNING"
      make -C lib
      export CC="${gfortran}/bin/gcc";
      '';

      propagatedBuildInputs = [
          mpi4py
          numpy
          pyyaml
          netcdf4
          fv3gfs-util
      ];
  }