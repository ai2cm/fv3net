# Copied from https://nixos.org/nixos/nix-pills/callpackage-design-pattern.html
let
  # nixpkgs = import <nixpkgs> {};
  # allPkgs = nixpkgs // pkgs;
  # callPackage = path: overrides:
  #   let f = import path;
  #   in f ((builtins.intersectAttrs (builtins.functionArgs f) allPkgs) // overrides);
  packageOverrides = mpi: fv3: self: super: rec {
    wrapper = super.callPackage ./wrapper.nix {inherit fv3;};
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
    fv3gfs-util = super.buildPythonPackage {
      version = "0.1.0";
      pname = "fv3gfs-wrapper";
      src = ../external/fv3gfs-util;
      propagatedBuildInputs = [
        super.fsspec
        super.xarray
        super.zarr
        super.typing-extensions
        cftime
      ];

    };
  };
  overlay = self: super: with nixpkgs; rec {
    # ensure that everything uses mpich for mpi
    fms = super.callPackage ./fms { };
    esmf = super.callPackage ./esmf { };
    nceplibs = super.callPackage ./nceplibs { };
    fv3 = super.callPackage ./fv3 { };
    python3 = super.python3.override {
      packageOverrides = packageOverrides mpich fv3;
    };
    py = python.withPackages (ps: [ ps.pytest ps.pyyaml ps.wrapper]);
    wrapper = python.pkgs.wrapper;
  };
  nixpkgs = import <nixpkgs> {overlays = [overlay];};
  mypy = nixpkgs.python3.withPackages (ps: with ps; [pkgconfig jinja2 cython numpy mpi4py setuptools]);
  shell = with nixpkgs; mkShell {
  buildInputs = [
          mypy
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
          mpich
    ];

    MPI="mpich";
    FC="gfortran";
    inherit fms;
    CC="${gfortran}/bin/gcc";
    inherit gfortran;
    gfc="${gfortran.cc}";
};
in nixpkgs.python3Packages.wrapper