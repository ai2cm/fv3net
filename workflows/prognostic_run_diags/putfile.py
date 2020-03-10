import sys
import fsspec

fs = fsspec.filesystem('gs')
fs.put(*sys.argv)