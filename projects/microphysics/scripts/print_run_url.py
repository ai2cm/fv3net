import config
import argparse

from fv3net.artifacts.resolve_url import resolve_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="Prognostic run tag used to create the full url.")

    args = parser.parse_args()

    print(resolve_url(config.BUCKET, config.PROJECT, args.tag))
