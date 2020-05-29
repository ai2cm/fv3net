import argparse
import yaml
import os
import sherpa

TRIAL_PY_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trial.py")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", help="configuration yaml location", type=str)
    parser.add_argument(
        "outdir", help="output location for scheduler and database files", type=str
    )
    parser.add_argument(
        "--max_concurrent", help="Number of concurrent processes", type=int, default=1
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.configfile, "r") as f:
        config = yaml.safe_load(f)
    algorithm_name = config["algorithm"]["name"]
    if not hasattr(sherpa.algorithms, algorithm_name):
        raise ValueError(
            f"No sherpa algorithm {algorithm_name} exists, is there a typo?"
        )
    algorithm = getattr(sherpa.algorithms, algorithm_name)(
        *config["algorithm"].get("args", []), **config["algorithm"].get("kwargs", {})
    )
    parameters = []
    for parameter_config in config["parameters"]:
        parameters.append(sherpa.core.Parameter.from_dict(parameter_config))
    scheduler = sherpa.schedulers.LocalScheduler()
    return_value = sherpa.optimize(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=True,
        filename=TRIAL_PY_FILENAME,
        output_dir=args.outdir,
        max_concurrent=args.max_concurrent,
        scheduler=scheduler,
        db_port=27001,
    )
    print(f"Best results: {return_value}")
