import argparse
from utils import create_experiment_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow-config",
        type=str,
        required=True,
        help="Location of workflow config yaml."
    )
    
    args = parser.parse_args()
    
    # run the function
    create_experiment_path(args)