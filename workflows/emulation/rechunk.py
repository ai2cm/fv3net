import zarr
import rechunker
import argparse
import logging
import glob
from functools import partial
from tempfile import TemporaryDirectory
from pathlib import Path
from dask.diagnostics import ProgressBar

logger = logging.getLogger(__name__)
MAX_MEM = "256MB"


def _in_mb(nbytes_per_item, size):
    return size * nbytes_per_item // 1024 ** 2


def get_target_chunk(data, target_size_mb):

    chunks = []
    shape = data.shape
    itemsize = data.dtype.itemsize
    get_mb = partial(_in_mb, itemsize)

    curr_size = 1
    for dim_size in shape[::-1]:

        if get_mb(curr_size) >= target_size_mb:
            chunks.insert(0, 1)
        else:
            for i in range(1, dim_size + 1):
                if get_mb(curr_size * i) >= target_size_mb:
                    chunks.insert(0, i)
                    curr_size *= i
                    break
            else:
                chunks.insert(0, dim_size)
                curr_size *= dim_size

    return chunks


def rechunk_dataset(input_path, output_path, chunksize_mb):

    logger.info(f"Rechunking zarr at path: {input_path} to {output_path}")
    dataset = zarr.open(input_path)
    with TemporaryDirectory() as tmpdir:
        filename = Path(input_path).name
        tmp_zarr = Path(tmpdir, filename).as_posix()
        chunks = {var: get_target_chunk(dataset[var], chunksize_mb) for var in dataset}
        plan = rechunker.rechunk(
            dataset, target_chunks=chunks, 
            max_mem=MAX_MEM, target_store=output_path, temp_store=tmp_zarr
        )
        with ProgressBar():
            plan.execute()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing zarr files to rechunk.")
    parser.add_argument(
        "output_dir",
        help=(
            "Directory to place rechunked files. Note: should be different "
            "than input to prevent file conflicts."
        ),
    )
    parser.add_argument("--chunksize_mb", required=False, default=5, type=int)

    args = parser.parse_args()

    zarr_paths = glob.glob(Path(args.input_dir, "*.zarr").as_posix())
    for path in zarr_paths:
        in_zarr = Path(path)
        out_zarr = Path(args.output_dir, in_zarr.name)
        rechunk_dataset(in_zarr.as_posix(), out_zarr.as_posix(), args.chunksize_mb)
