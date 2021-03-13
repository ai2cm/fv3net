# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, fetchPypi, lib, importlib-resources, matplotlib, numpy, pandas, pint, pooch
, pyproj, scipy, setuptools_scm, traitlets, xarray }:

buildPythonPackage rec {
  pname = "metpy";
  version = "1.0";

  src = fetchPypi {
    inherit version;
    pname = "MetPy";
    sha256 = "1p00n44830rgssibqmg1afk3vxxrgsbbjsz936rmvlz3ljm47c0i";
  };

  buildInputs = [ setuptools_scm ];
  propagatedBuildInputs =
    [ importlib-resources matplotlib numpy pandas pint pooch pyproj scipy traitlets xarray ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; { };
}
