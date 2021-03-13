# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, fetchPypi, lib }:

buildPythonPackage rec {
  pname = "bump2version";
  version = "1.0.1";

  src = fetchPypi {
    inherit pname version;
    sha256 = "1rinm4gv1fyh7xjv3v6r1p3zh5kl4ry2qifz5f7frx31mnzv4b3n";
  };

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description = "Version-bump your software with a single command!";
    homepage = "https://github.com/c4urself/bump2version";
  };
}
