# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, fetchPypi, lib, pytest }:

buildPythonPackage rec {
  pname = "pytest-regtest";
  version = "1.4.5";

  src = fetchPypi {
    inherit pname version;
    sha256 = "1qw6dwzfpa27dw3ai7xpm9ncpb7ddnj72nniysdksj9y91yyir51";
  };

  propagatedBuildInputs = [ pytest ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description = "pytest plugin for regression tests";
    homepage = "https://gitlab.com/uweschmitt/pytest-regtest";
  };
}
