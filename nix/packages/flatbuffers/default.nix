# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, fetchPypi, lib }:

buildPythonPackage rec {
  pname = "flatbuffers";
  version = "20210313004315";

  # TODO use fetchPypi
  src = builtins.fetchurl {
    url =
      "https://files.pythonhosted.org/packages/4d/c4/7b995ab9bf0c7eaf10c386d29a03408dfcf72648df4102b1f18896c3aeea/flatbuffers-1.12.tar.gz";
    sha256 = "0426nirqv8wzj56ppk6m0316lvwan8sn28iyj40kfdsy5mr9mfv3";
  };

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description = "The FlatBuffers serialization format for Python";
    homepage = "https://google.github.io/flatbuffers/";
  };
}
