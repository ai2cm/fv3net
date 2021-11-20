import xarray as xr
import fsspec
import click

URL_IN = (
    "gs://vcm-ml-experiments/default/2021-04-27/"
    "2020-05-27-40-day-X-SHiELD-simulation/fine-res-budget.zarr"
)
URL_OUT = (
    "gs://vcm-ml-intermediate/2021-11-18-fine-res-budget-"
    "rechunked-from-2020-05-27-40-day-X-SHiELD-simulation.zarr"
)
TARGET_CHUNKS = [("time", 24), ("tile", 1)]


@click.command()
@click.argument("url_in", type=str, default=URL_IN)
@click.argument("url_out", type=str, default=URL_OUT)
@click.option("--chunk", type=(str, int), multiple=True, default=TARGET_CHUNKS)
def rechunk(url_in, url_out, chunk):
    chunks_dict = {k: v for k, v in chunk}
    ds = xr.open_zarr(fsspec.get_mapper(url_in))
    ds_rechunked = ds.chunk(chunks_dict)
    history = (
        f"Rechunked from {url_in} and saved to {url_out} using fv3net/projects"
        "/fine-res/scripts/rechunk_fine_res.py"
    )
    ds_rechunked.attrs["history"] = history
    ds_rechunked.to_zarr(fsspec.get_mapper(url_out), mode="w", consolidated=True)


if __name__ == "__main__":
    rechunk()
