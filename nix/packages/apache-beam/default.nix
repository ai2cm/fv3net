# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, fetchPypi, lib }:

buildPythonPackage rec {
  pname = "apache-beam";
  version = "2.28.0";

  src = fetchPypi {
    inherit pname version;
    extension = "zip";
    sha256 = "1wvfhn2k9dp5mrdb64fv8fssdblg4rjam8hz0bl966nbfl6mdl9b";
  };

  # TODO FIXME
  doCheck = false;

  meta = with lib; { };
}
