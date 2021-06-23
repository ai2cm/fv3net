# TODO delete this dev script
set -e

pytest runtime/emulator
guild run --tag dev -y epochs=1 nfiles=1 problem=single-level lr=0.0001 wandb_logger=False
guild run --tag dev -y epochs=1 nfiles=1 problem=all lr=0.0001 wandb_logger=False
guild run --tag dev -y epochs=1 nfiles=1 problem=rh lr=10.0 wandb_logger=False scale=1.0
