import argparse
import yaml

from fv3fit._shared import put_dir
from fv3fit.reservoir.adapters import generate_subdomain_models_from_model_map


def main(model_map_file, output_dir):
    with open(model_map_file) as f:
        model_map = yaml.safe_load(f)

    with put_dir(output_dir) as tmpdir:
        new_model_map = generate_subdomain_models_from_model_map(model_map, tmpdir)
        new_model_map = {
            k: v.replace(f"{tmpdir}", f"{output_dir}") for k, v in new_model_map.items()
        }
        print(yaml.dump(new_model_map, indent=2))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "model_map_file", help="yaml file mapping rank index to reservoir model path"
    )
    argparser.add_argument(
        "output_dir", help="output directory to save subdomain models"
    )

    args = argparser.parse_args()

    main(args.model_map_file, args.output_dir)
