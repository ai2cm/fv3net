import json
import argparse
import re

# Read the list of template parameters from the JSON file
with open("template_parameters.json", "r") as file:
    template_param_names = set(json.load(file))

# Set up the argument parser
parser = argparse.ArgumentParser(description="Validate Argo parameters.")
parser.add_argument(
    "-p",
    "--parameter",
    action="append",
    dest="parameters",
    metavar="PARAM",
    help="Specify a parameter with its value",
    default=[],
)

# Parse the submission arguments
args, _ = parser.parse_known_args()

# Extract provided parameters
provided_params = {re.match(r"([\w-]+)", param).group(1) for param in args.parameters}

# Compare template and provided parameters
unused_params = provided_params - template_param_names
unspecified_params = template_param_names - provided_params

if unspecified_params:
    print("Warning: Some parameters are unspecified: ", ", ".join(unspecified_params))

if unused_params:
    print("Error: Some parameters are unused or misspelled: ", ", ".join(unused_params))
    exit(1)
