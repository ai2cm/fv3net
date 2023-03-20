import pytest
from ..save_ranks import _get_parser

tstart = "20200121.103000"
tstop = "20200122.000000"
tchunks = "10"


@pytest.mark.parametrize(
    "kwargs",
    [
        [
            "--start-time",
            tstart,
            "--stop-time",
            tstop,
            "--time-chunks",
            tchunks,
            "--variables",
            "T",
            "q",
        ],
        [
            "--variables",
            "T",
            "q",
            "--start-time",
            tstart,
            "--stop-time",
            tstop,
            "--time-chunks",
            tchunks,
        ],
        ["--variables", "T", "q", "--time-chunks", tchunks],
    ],
)
def test_arg_parser_variables_list(kwargs):
    positional_args = ["gs://data_path", "gs://output_path", "4", "2"]
    parser = _get_parser()
    args = parser.parse_args(positional_args + kwargs)
    assert args.variables == ["T", "q"]
