# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, cftime, fetchPypi, lib, matplotlib, numpy, six }:

buildPythonPackage rec {
  pname = "nc-time-axis";
  version = "1.2.0";

  src = fetchPypi {
    inherit pname version;
    sha256 = "1h9331yynz3hvmmfnb6algmdrcvzvrhzjygn2npmrs0cyhzndks9";
  };

  propagatedBuildInputs = [ cftime matplotlib numpy six ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description = "cftime support for matplotlib axis";
    homepage = "https://github.com/scitools/nc-time-axis";
  };
}
