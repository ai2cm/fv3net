# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, fetchPypi, lib, google_auth
    , google-auth-oauthlib
    , requests
    , decorator
    , fsspec
    , aiohttp
    , ujson }:

buildPythonPackage rec {
  pname = "gcsfs";
  version = "0.7.2";

  src = fetchPypi {
    inherit pname version;
    sha256 = "1pm475s71433mkgajvykj74jra4llw1syibs3yrpxkfk58s8whz4";
  };

  propagatedBuildInputs = [
    google_auth
    google-auth-oauthlib
    requests
    decorator
    fsspec
    aiohttp
    ujson
  ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; { };
}
