# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, fetchPypi, lib, sphinx }:

buildPythonPackage rec {
  pname = "sphinx-gallery";
  version = "0.8.2";

  src = fetchPypi {
    inherit pname version;
    sha256 = "0qj4vq73dzqncw89fv1jg0f4pfl3cvsjgjdi9dkw335qz4dbmzkl";
  };

  propagatedBuildInputs = [ sphinx ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description =
      "A Sphinx extension that builds an HTML version of any Python script and puts it into an examples gallery.";
    homepage = "https://sphinx-gallery.github.io";
  };
}
