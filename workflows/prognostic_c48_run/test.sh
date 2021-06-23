# TODO delete this dev script
set -e

pytest tests/test_thermo.py tests/test_loss.py tests/test_emulator.py 
guild run --tag dev epochs=5 nfiles=5 problem=single-level lr=0.0001 wandb_logger=False

