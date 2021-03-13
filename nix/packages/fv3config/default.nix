# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ appdirs, backoff, buildPythonPackage, dacite, f90nml, fetchPypi, fsspec, gcsfs
, lib, pyyaml, requests }:

buildPythonPackage rec {
  pname = "fv3config";
  version = "0.6.1";

  src = fetchPypi {
    inherit pname version;
    sha256 = "00f0c1c1lnbhwmyg1kjvp0yxbj1kj3pls7jh2zywb0l0fs72n2px";
  };

  propagatedBuildInputs =
    [ f90nml appdirs requests pyyaml gcsfs fsspec backoff dacite ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description =
      "FV3Config is used to configure and manipulate run directories for FV3GFS.";
    homepage = "https://github.com/VulcanClimateModeling/fv3config";
  };
}
