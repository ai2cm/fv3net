# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, click, click-default-group, ensureconda, fetchPypi, jinja2
, lib, pyyaml, requests, setuptools, setuptools_scm, toml }:

buildPythonPackage rec {
  pname = "conda-lock";
  version = "0.8.0";

  src = fetchPypi {
    inherit version;
    pname = "conda_lock";
    sha256 = "18gy842hkfm2frs47sl5lvnxisgkldyvlkkp5x164hbshz09xacz";
  };

  buildInputs = [ setuptools setuptools_scm ];
  propagatedBuildInputs =
    [ pyyaml requests jinja2 toml ensureconda click click-default-group ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; { };
}
