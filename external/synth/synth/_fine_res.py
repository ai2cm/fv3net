from typing import Sequence
from synth.schemas import load_schema
from synth.core import generate
import os


def generate_fine_res(datadir: str, times: Sequence[str]):
    """Generate a directory of fine-res data
    
    Args:
        datadir: output location
        times: list of YYYYMMDD.HHMMSS timestamps
    
    """
    schema = load_schema("fine_res_budget.json")
    dataset = generate(schema)

    for tile in range(1, 7):
        for time in times:
            path = os.path.join(datadir, f"{time}.tile{tile}.nc")
            dataset.isel(tile=tile-1).to_netcdf(path)
