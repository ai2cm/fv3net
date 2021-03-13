# WARNING: This file was automatically generated. You should avoid editing it.
# If you run pynixify again, the file will be either overwritten or
# deleted, and you will lose the changes you made to it.

{ buildPythonPackage, dask, docrep, fetchPypi, future, lib, numpy, xarray }:

buildPythonPackage rec {
  pname = "xgcm";
  version = "0.5.1";

  src = fetchPypi {
    inherit pname version;
    sha256 = "1cq8lfiy314yrwniw2c2ppgzazqf5agi9yd7l6dj8n72pinyn8vr";
  };

  propagatedBuildInputs = [ xarray dask numpy future docrep ];

  # TODO FIXME
  doCheck = false;

  meta = with lib; {
    description = "General Circulation Model Postprocessing with xarray";
    homepage = "https://github.com/xgcm/xgcm";
  };
}
