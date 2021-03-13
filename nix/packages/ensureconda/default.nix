# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ appdirs, buildPythonPackage, click, fetchPypi, filelock, lib, requests
, setuptools, setuptools_scm }:

buildPythonPackage rec {
  pname = "ensureconda";
  version = "1.4.1";

  src = fetchPypi {
    inherit pname version;
    sha256 = "1av2xm7gnbdsd0wcq1hxzjmq0pxxwaa2hqnhvbzr0axpp2daslhc";
  };

  buildInputs = [ setuptools setuptools_scm ];
  propagatedBuildInputs = [ click requests appdirs filelock ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; { };
}
