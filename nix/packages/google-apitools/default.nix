# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, fasteners, fetchPypi, httplib2, lib, oauth2client, six }:

buildPythonPackage rec {
  pname = "google-apitools";
  version = "0.5.31";

  src = fetchPypi {
    inherit pname version;
    sha256 = "0w21c83qjk44jp0mnk55znnkv1y1jxxbbw2i09li0a2qsinxvw2a";
  };

  propagatedBuildInputs = [ httplib2 fasteners oauth2client six ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description = "client libraries for humans";
    homepage = "http://github.com/google/apitools";
  };
}
