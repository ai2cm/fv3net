import subprocess
import re
import typing
from runtime.segmented_run.logs import (
    LogLine,
    LineType,
    parse_line,
    handle_fv3_log,
    floating_point,
)


def test_parse_line():
    assert LogLine(
        LineType.FV3_LOG, {"message": "some data"}, "some data"
    ) == parse_line("some data")
    line = "INFO:fv3:[1]"
    return LogLine(
        LineType.PYTHON_LOG, {"severity": "INFO", "module": "fv3", "json": [1]}, line
    ) == parse_line(line)


def test_parse_floating_point():
    assert re.search(floating_point, "1.0E-002").group(0) == "1.0E-002"
    assert re.search(floating_point, "2.0").group(0) == "2.0"
    assert re.search(floating_point, "-123.0").group(0) == "-123.0"


def test_Handler(regtest: typing.TextIO):
    data = """DEBUG:runtime.loop:Physics Step (apply)
WARNING from PE     0: diag_util_mod::opening_file: module/field_name (dynamics/w50) NOT registered
 PS max =    1034.9816787407742       min =    539.31719196478946
WARNING from PE     0: diag_util_mod::opening_file: module/field_name (dynamics/w50) NOT registered
 ---isec,seconds         900        1800
  gfs diags time since last bucket empty:   0.25000000000000000      hrs
INFO:runtime.loop:Computing Postphysics Updates
INFO:runtime.loop:Computing Postphysics Updates
INFO:runtime.loop:{"a": 1}
    """  # noqa
    input_stream = data.splitlines()
    for line in handle_fv3_log(input_stream):
        print(line, file=regtest)


def test_read_from_subprocess():
    """Check that we can iterate over lines from a subprocess as expected"""
    proc = subprocess.Popen(["printf", "a\nb\nc\nd"], stdout=subprocess.PIPE, text=True)
    assert list(proc.stdout) == ["a\n", "b\n", "c\n", "d"]
