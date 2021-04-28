from synth.core import Range, write_directory_schema
from . import schemas


def generate_nudging(outdir: str):
    ranges = {"pressure_thickness_of_atmospheric_layer": Range(0.99, 1.01)}
    directory_schema = schemas.load_directory_schema_from_library("nudge_to_fine")
    write_directory_schema(outdir, directory_schema, ranges)
