import os
from post_process import parse_rundir


def test_parse_rundir_mocked_walker():
    walker = [
        (
            ".",
            ["diags.zarr", "INPUT", "OUTPUT"],
            ["a.tile1.nc", "a.tile2.nc", "a.tile3.nc", "randomfile"],
        ),
        ("./diags.zarr", ["a"], [".zattrs"],),
        ("./diags.zarr/a", [], ["0", "1", ".zattrs"],),
        ("./INPUT", [], ["restart.nc"],),
    ]
    tiles, zarrs, other = parse_rundir(walker)

    assert tiles == ["./a.tile1.nc", "./a.tile2.nc", "./a.tile3.nc"]
    assert zarrs == ["./diags.zarr"]
    assert set(other) == {"./INPUT/restart.nc", "./randomfile"}


def test_parse_rundir_os_walk_integration(tmpdir):
    # Setup directory structure
    zarr = tmpdir.mkdir("diags.zarr")
    input_ = tmpdir.mkdir("INPUT")

    zarr.join("0").write("")
    zarr.join("1").write("")
    tmpdir.join("a.tile1.nc").write("")
    tmpdir.join("a.tile2.nc").write("")
    tmpdir.join("a.tile3.nc").write("")
    tmpdir.join("randomfile").write("")

    input_.join("restart.nc").write("")

    tiles, zarrs, other = parse_rundir(os.walk(str(tmpdir)))

    assert set(tiles) == {
        f"{tmpdir}/a.tile1.nc",
        f"{tmpdir}/a.tile2.nc",
        f"{tmpdir}/a.tile3.nc",
    }
    assert set(zarrs) == {f"{tmpdir}/diags.zarr"}
    assert set(other) == {f"{tmpdir}/INPUT/restart.nc", f"{tmpdir}/randomfile"}
