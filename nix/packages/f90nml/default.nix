# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, fetchPypi, lib }:

buildPythonPackage rec {
  pname = "f90nml";
  version = "1.2";

  src = fetchPypi {
    inherit pname version;
    sha256 = "0bndjjk9xw8nhg6x36mn1bnwkv8n2744w9ilrjffi3113w8bkyq7";
  };

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description = "Fortran 90 namelist parser";
    homepage = "http://github.com/marshallward/f90nml";
  };
}
