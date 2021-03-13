# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, fetchPypi, lib, setuptools }:

buildPythonPackage rec {
  pname = "fastavro";
  version = "1.3.2";

  src = fetchPypi {
    inherit pname version;
    sha256 = "1f0as7s3qw3ay077005ccxx99sd3jfa538psbq7rwfwgc99fqnv6";
  };

  buildInputs = [ setuptools ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description = "Fast read/write of AVRO files";
    homepage = "https://github.com/fastavro/fastavro";
  };
}
