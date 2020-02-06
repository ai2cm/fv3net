import argparse
from utils import get_step_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="Path of the experiment root directory."
    )
    parser.add_argument(
        "--workflow-config",
        type=str,
        required=True,
        help="Location of workflow config yaml."
    )
    parser.add_argument(
        "--workflow-step",
        type=str,
        required=True,
        help="Step in the workflow for which to generate inputs."
    )
    args = parser.parse_args()
    
    # run the function
    get_step_args(args)
