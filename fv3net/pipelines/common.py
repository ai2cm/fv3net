import apache_beam as beam
from apache_beam import io
import xarray as xr
import tempfile
import shutil
import os


class CombineSubtilesByKey(beam.PTransform):
    def expand(self, pcoll):
        return pcoll | beam.GroupByKey() | beam.MapTuple(self._combine)

    @staticmethod
    def _combine(key, datasets):
        return key, xr.combine_by_coords(datasets)


class _NetCDFSink(io.FileBasedSink):
    def __init__(self):
        super(_NetCDFSink, self).__init__(file_path_prefix="", coder=None)

    def write_record(self, file_handle, value: xr.Dataset):
        value.to_netcdf(file_handle)


class WriteToNetCDFs(beam.PTransform):
    def __init__(self, name_fn):
        self._sink = _NetCDFSink()
        self.name_fn = name_fn

    def _process(self, key, elm: xr.Dataset):
        path = self.name_fn(key)
        dest = self._sink.open(path)
        tmp = tempfile.mktemp()

        try:
            elm.to_netcdf(tmp)
            with open(tmp, "rb") as src:
                shutil.copyfileobj(src, dest)
        finally:
            dest.close()
            os.unlink(tmp)

    def expand(self, pcoll):
        return pcoll | beam.MapTuple(self._process)
