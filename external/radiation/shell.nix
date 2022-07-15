# shell.nix
let
  pkgs = import (builtins.fetchGit {
    # Descriptive name to make the store path easier to identify
    name = "ai2cm-packages";
    url = "git@github.com:VulcanClimateModeling/packages.git";
    ref = "master";
    # SHA of the commit of pkgs to use. This effectively pins all packages to
    # the versions specified in that commit.
    # rev = "b3194fd82e98c288d04ca6fdd1091b484371b26c";
    rev = "7f6bd8ccb8217fc6012aa287d8e61474ee4d46e3";
  }) { };
in pkgs.mkShell {
  buildInputs = [
    pkgs.python38Packages.numba
    pkgs.python38Packages.xarray
    pkgs.python38Packages.numpy
    pkgs.python38Packages.netcdf4
    pkgs.python38Packages.ipython

  ];
  shellHook = ''
    export PYTHONPATH=${pkgs.serialbox}/python:$PYTHONPATH
  '';
}
