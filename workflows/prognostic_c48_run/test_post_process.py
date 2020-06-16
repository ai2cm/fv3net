import os
import numpy as np
import xarray as xr
from post_process import parse_rundir, process_item
import tempfile


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


def test_process_item_dataset(tmpdir):
    d_in = str(tmpdir)
    localpath = str(tmpdir.join("diags.zarr"))
    ds = xr.Dataset(
        {"a": (["time", "x"], np.ones((200, 10)))}, attrs={"path": localpath}
    )
    with tempfile.TemporaryDirectory() as d_out:
        process_item(ds, d_in, d_out)
        xr.open_zarr(d_out + "/diags.zarr")


def test_process_item_str(tmpdir):
    txt = "hello"
    d_in = str(tmpdir)
    path = tmpdir.join("afile.txt")
    path.write(txt)

    with tempfile.TemporaryDirectory() as d_out:
        process_item(str(path), d_in, d_out)
        with open(d_out + "/afile.txt") as f:
            assert f.read() == txt


def test_process_item_str_nested(tmpdir):
    txt = "hello"
    d_in = str(tmpdir)
    path = tmpdir.mkdir("nest").join("afile.txt")
    path.write(txt)

    with tempfile.TemporaryDirectory() as d_out:
        process_item(str(path), d_in, d_out)
        with open(d_out + "/nest/afile.txt") as f:
            assert f.read() == txt


def test_process_item_broken_symlink(tmpdir):
    fake_path = str(tmpdir.join("idontexist"))
    broken_link = str(tmpdir.join("broken_link"))
    os.symlink(fake_path, broken_link)
    with tempfile.TemporaryDirectory() as d_out:
        process_item(broken_link, str(tmpdir), d_out)
