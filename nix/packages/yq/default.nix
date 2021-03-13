# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ argcomplete, buildPythonPackage, fetchPypi, lib, pyyaml, setuptools, toml
, xmltodict }:

buildPythonPackage rec {
  pname = "yq";
  version = "2.12.0";

  src = fetchPypi {
    inherit pname version;
    sha256 = "082ciixnl6k9fnwa45xvnxcavmsnk27njv5qb196nc2da01x8ahx";
  };

  propagatedBuildInputs = [ setuptools pyyaml xmltodict toml argcomplete ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description =
      "Command-line YAML/XML processor - jq wrapper for YAML/XML documents";
    homepage = "https://github.com/kislyuk/yq";
  };
}
