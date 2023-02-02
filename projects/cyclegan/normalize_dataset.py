import xarray as xr
import sklearn.preprocessing
import vcm

VARNAMES = [
    "h500",
    "PRATEsfc",
    "PRATEsfc_log",
    "PRESsfc",
    "w500",
    "TMPlowest",
    "TMP500_300",
    "PWAT",
]


def _normalize_ds(ds):
    ds = ds.drop_vars([var for var in ds.data_vars if var not in VARNAMES])
    # normalize dataset using sklearn.preprocessing.QuantileTransformer
    for var in VARNAMES:
        # reshape data to 2D array
        data = ds[var].values.reshape(-1, 1)
        # fit and transform data
        scaler = sklearn.preprocessing.QuantileTransformer(output_distribution="normal")
        scaler.fit(data)
        data = scaler.transform(data)
        # reshape data back to original shape
        data = data.reshape(ds[var].shape)
        # assign normalized data to dataset
        ds[var][:] = data
    return ds


def normalize_dataset(input_path: str, output_path: str):
    # Open input dataset
    ds: xr.Dataset = xr.open_zarr(input_path)
    if vcm.get_fs(output_path).exists(output_path):
        print(f"Output path {output_path} already exists. Skipping.")
    else:
        # Normalize dataset
        ds = _normalize_ds(ds)
        # Save normalized dataset
        ds.to_zarr(output_path)


if __name__ == "__main__":
    # Parse command line arguments
    # program takes an input path and an output path as required positionalarguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input_path", type=str)
    # parser.add_argument("output_path", type=str)
    # args = parser.parse_args()

    # Normalize dataset
    normalize_dataset(
        "gs://vcm-ml-experiments/mcgibbon/2023-01-11/coarse-0K.zarr/",
        "gs://vcm-ml-experiments/mcgibbon/2023-01-11/coarse-0K-norm.zarr/",
    )
    normalize_dataset(
        "gs://vcm-ml-experiments/mcgibbon/2023-01-11/fine-0K.zarr/",
        "gs://vcm-ml-experiments/mcgibbon/2023-01-11/fine-0K-norm.zarr/",
    )
    ds = xr.open_zarr(
        "gs://vcm-ml-experiments/mcgibbon/2023-01-11/coarse-0K-norm.zarr/"
    )
