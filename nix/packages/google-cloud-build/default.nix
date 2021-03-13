# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, fetchPypi, google_api_core, lib, libcst, proto-plus }:

buildPythonPackage rec {
  pname = "google-cloud-build";
  version = "3.0.2";

  src = fetchPypi {
    inherit pname version;
    sha256 = "1ryhlyr8lfdwwsf1ackz3pi4br5s314j6c10wxjhds4p9797z4kq";
  };

  propagatedBuildInputs = [ google_api_core proto-plus libcst ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description = "Google Cloud Build API client library";
    homepage = "https://github.com/googleapis/python-cloudbuild";
  };
}
