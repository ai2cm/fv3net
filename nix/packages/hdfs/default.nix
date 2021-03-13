# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, docopt, fetchPypi, lib, requests, six }:

buildPythonPackage rec {
  pname = "hdfs";
  version = "2.6.0";

  src = fetchPypi {
    inherit pname version;
    sha256 = "08nw3jnyjfnnzdzlgv9yzn69jxh4jfm6qx8zaj5x81pi8x1wx4mw";
  };

  propagatedBuildInputs = [ docopt requests six ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description = "HdfsCLI: API and command line interface for HDFS.";
    homepage = "https://hdfscli.readthedocs.io";
  };
}
