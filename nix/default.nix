# Copied from https://nixos.org/nixos/nix-pills/callpackage-design-pattern.html
let
  # nixpkgs = import <nixpkgs> {};
  # allPkgs = nixpkgs // pkgs;
  # callPackage = path: overrides:
  #   let f = import path;
  #   in f ((builtins.intersectAttrs (builtins.functionArgs f) allPkgs) // overrides);
  packageOverrides = nixpkgs: mpi: fv3: self: super: rec {

    apache-beam = self.callPackage ./packages/apache-beam { };

    bump2version = self.callPackage ./packages/bump2version { };

    conda-lock = self.callPackage ./packages/conda-lock { };

    dacite = self.callPackage ./packages/dacite { };

    ensureconda = self.callPackage ./packages/ensureconda { };

    f90nml = self.callPackage ./packages/f90nml { };

    fastavro = self.callPackage ./packages/fastavro { };

    flatbuffers = self.callPackage ./packages/flatbuffers { };

    fv3config = self.callPackage ./packages/fv3config { };

    gcsfs = self.callPackage ./packages/gcsfs { };

    google-apitools = self.callPackage ./packages/google-apitools { };

    google-cloud-build = self.callPackage ./packages/google-cloud-build { };

    hdfs = self.callPackage ./packages/hdfs { };

    intake-xarray = self.callPackage ./packages/intake-xarray { };

    metpy = self.callPackage ./packages/metpy { };

    nc-time-axis = self.callPackage ./packages/nc-time-axis { };

    pytest-regtest = self.callPackage ./packages/pytest-regtest { };

    sphinx-gallery = self.callPackage ./packages/sphinx-gallery { };

    xgcm = self.callPackage ./packages/xgcm { };

    yq = self.callPackage ./packages/yq { };
    
    wrapper = self.callPackage ./wrapper.nix {inherit fv3;};

    # this test was very slow
    intake = super.intake.overrideAttrs (oldAttrs: {disabledTests = oldAttrs.disabledTests + ''"test_conf_auth"'';});

    # skimage 0.17.2 tries to write to a read-only directory
    # use lower version
    scikitimage = super.scikitimage.overrideAttrs (oldAttrs : rec {
      
      version = "0.16.2"; 
      pname = "scikit-image";

      src = self.fetchPypi {
        inherit pname version;
        sha256 = "13060l9vnaifw705s7z6rrk7jvpi67vlan61gnbfkm3lv8rbszyx";
      };
    
      });

    mpi4py = super.mpi4py.override {inherit mpi;};
    cftime = super.cftime.overrideAttrs (oldAttrs: {
        pname = "cftime";
        version="1.2.1";
        src = super.fetchPypi {
          pname = "cftime";
          version="1.2.1";
          sha256 = "ab5d5076f7d3e699758a244ada7c66da96bae36e22b9e351ce0ececc36f0a57f";
        };
      });
    numcodecs = super.numcodecs.overrideAttrs (oldAttrs: {
      disabledTests = [
        "test_backwards_compatibility"
        "test_encode_decode"
        "test_legacy_codec_broken"
        "test_bytes"
      ];
    });

    fv3gfs-util = self.buildPythonPackage {
      version = "0.1.0";
      pname = "fv3gfs-wrapper";
      src = ../external/fv3gfs-util;
      propagatedBuildInputs = with self; [
        fsspec
        xarray
        zarr
        typing-extensions
        cftime
      ];
    };

    fv3kube = self.buildPythonPackage {
          pname = "fv3kube";
          version = "na";
          src = ../external/fv3kube;

          FV3CONFIG_CACHE_DIR = ".";
          checkInputs = [ self.pytestCheckHook pytest-regtest ];
          propagatedBuildInputs = with self; [
            kubernetes
            fv3config
          ];
        };

    vcm = self.buildPythonPackage {
          pname = "fv3kube";
          version = "na";
          src = ../external/vcm;

        doCheck = false;
          checkInputs = [  pytest-regtest ];
          nativeBuildInputs = [ 
            nixpkgs.gfortran
          ];
          propagatedBuildInputs = with self; [
            click
            f90nml
            appdirs
            requests
            h5py
            dask
            xarray
            toolz
            scipy
            scikitimage
            metpy
            pooch
            intake
            gcsfs
            zarr
            xgcm
            cftime
            google_cloud_storage
            google_api_core
            h5netcdf
            docrep
            pytest-regtest
            intake-xarray
          ];
        };

  };
  overlay = self: super: with nixpkgs; rec {
    # ensure that everything uses mpich for mpi

    fms = self.callPackage ./fms { };
    esmf = self.callPackage ./esmf { };
    nceplibs = self.callPackage ./nceplibs { };
    fv3 = self.callPackage ./fv3 { };
    python3 = super.python3.override {
      packageOverrides = packageOverrides self mpich fv3;
    };
    wrapper = python.pkgs.wrapper;
  };
  nixpkgs = import (builtins.fetchTarball {
  # Descriptive name to make the store path easier to identify
  name = "release-20.09";
  # Commit hash for nixos-unstable as of 2018-09-12
  url = "https://github.com/nixos/nixpkgs/archive/20.09.tar.gz";
  # Hash obtained using `nix-prefetch-url --unpack <url>`
  sha256 = "1wg61h4gndm3vcprdcg7rc4s1v3jkm5xd7lw8r2f67w502y94gcy";
}) {overlays = [overlay];};
  loaders_reqs = (ps: with ps; [joblib intake]);
  fv3fit_reqs = (ps: with ps; []);
  py = nixpkgs.python3.withPackages (ps: with ps; [ pip setuptools numpy wrapper fv3config pytest dask scikitimage xgcm metpy tox fv3kube vcm] ++ loaders_reqs ps ++ fv3fit_reqs ps);
  shell = with nixpkgs; mkShell {
  buildInputs = [
          pkg-config
          fv3
          fms
          esmf
          nceplibs
          netcdf
          netcdffortran
          lapack
          blas
          gfortran 
          # see this issue: https://discourse.nixos.org/t/building-shared-libraries-with-fortran/11876/2
          gfortran.cc.lib
          llvmPackages.openmp
          mpich
          py
          # debugging
          gdb

          # for cartopy
          geos 
          proj_5
    ] ++ nixpkgs.python3.pkgs.tensorflow_2.propagatedBuildInputs;

    MPI="mpich";
    FC="gfortran";
    inherit fms;
    CC="${gfortran}/bin/gcc";
    inherit gfortran;
    gfc="${gfortran.cc}";
    cclib="${gfortran.cc.lib}";

  shellHook = ''
    # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
    # See https://pip.pypa.io/en/stable/user_guide/#environment-variables.
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH

    # export PIP_CONSTRAINT=$(pwd)/constraints.txt
    export PIP_NO_BINARY=shapely,cartopy,mpi4py
    export GEOS_CONFIG=geos-config

    export LD_LIBRARY_PATH="$cclib/lib"
    # need this for shapely to work
    export DYLD_LIBRARY_PATH=${geos}/lib
    export CC=${gfortran.cc}/bin/gcc

    # local packages
    export PYTHONPATH=$(pwd)/external/vcm:$(pwd)/external/loaders:$(pwd)/external/synth:$(pwd)/external/fv3fit:$PYTHONPATH


    # source .env/bin/activate
  '';
};
in shell
