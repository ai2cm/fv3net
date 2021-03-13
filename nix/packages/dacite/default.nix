# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, fetchPypi, lib }:

buildPythonPackage rec {
  pname = "dacite";
  version = "1.6.0";

  src = fetchPypi {
    inherit pname version;
    sha256 = "1px6m9yq82mmj3q9hx4x7xgz92400f0gjfs9kzgd6lh31bnjb0fl";
  };

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description = "Simple creation of data classes from dictionaries.";
    homepage = "https://github.com/konradhalas/dacite";
  };
}
