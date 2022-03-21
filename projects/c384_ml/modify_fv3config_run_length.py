import fsspec
import os
import yaml
import logging
import argparse

logging.basicConfig(level=logging.INFO)

CONFIG_NAME = "fv3config.yml"


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_url",
        type=str,
        help=(
            "URL of FV3GFS run to modify. Follow with keyword args of coupler_nml "
            "changes, e.g., --hours=3"
        ),
    )
    return parser


def main(args, unknown_args):

    run_url = args.run_url
    new_run_length = {
        (arg.split("=")[0]).strip("-"): int(arg.split("=")[1]) for arg in unknown_args
    }

    logging.info(f"Opening run URL at {run_url} to modify run length.")

    fs, _, _ = fsspec.get_fs_token_paths(run_url)
    config = yaml.safe_load(fs.cat(os.path.join(run_url, CONFIG_NAME)))

    for k, v in new_run_length.items():
        logging.info(
            f"Updating from the following in fv3config.yaml: "
            f"{k} = {config['namelist']['coupler_nml'][k]}, "
            f"to the following: {k} = {v}."
        )
        config["namelist"]["coupler_nml"][k] = v

    with fs.open(os.path.join(run_url, CONFIG_NAME), "w") as f:
        f.write(yaml.dump(config))


if __name__ == "__main__":
    parser = _create_parser()
    args, unknown_args = parser.parse_known_args()
    main(args, unknown_args)
