import cmd
from fv3net.diagnostics.prognostic_run import load_run_data
import intake
import vcm.catalog
import vcm
import xarray as xr
import fv3viz
import pathlib
import matplotlib.pyplot as plt
import sys
import io
import warnings

from . import iterm

warnings.filterwarnings("ignore")


def meridional_transect(ds: xr.Dataset):
    transect_coords = vcm.select.meridional_ring()
    ds = vcm.interpolate_unstructured(ds, transect_coords)
    return ds.swap_dims({"sample": "lat"})


class PlotTape:
    def __init__(self):
        self.count = 0

    def save_plot(self):
        filename = f"image_{self.count}.png"
        plt.savefig(filename)
        plt.close(plt.gcf())
        self.count += 1


class OneFileTape:
    """Useful for working in vscode...updates file in place"""

    def save_plot(self):
        filename = f"image.png"
        plt.savefig(filename)
        plt.close(plt.gcf())


class ItermTape:
    width = 70

    def save_plot(self):
        f = io.BytesIO()
        plt.savefig(f)
        iterm.write_image(
            f.getvalue(),
            sys.stderr.buffer,
            filename="file",
            width=self.width,
            preserve_aspect_ratio=True,
        )
        plt.close(plt.gcf())


class State:
    def __init__(self):
        self.data_3d = None
        self.data_2d = None
        self.tape = OneFileTape()

    def load(self, url):
        prognostic = load_run_data.SegmentedRun(url, catalog)
        self.data_3d = prognostic.data_3d
        self.data_2d = prognostic.data_2d.merge(grid.drop("area"))

    def print(self):
        for v in self.data_3d:
            print(v)
        for v in self.data_2d:
            print(v)


catalog_path = vcm.catalog.catalog_path
catalog = intake.open_catalog(catalog_path)
grid = load_run_data.load_grid(catalog)

state = {}
loop_state = State()


def avg2d(state: State, variable):
    x = state.data_2d
    avg = vcm.weighted_average(x[variable], x.area, ["x", "y", "tile"])
    avg.plot()
    state.tape.save_plot()


def set_iterm_tape(state: State):
    state.tape = ItermTape()


def hovmoller(state: State, variable, vmin=None, vmax=None):
    z = state.data_2d[variable]
    avg = vcm.zonal_average_approximate(state.data_2d.lat, z)
    vmin = None if vmin is None else float(vmin)
    vmax = None if vmax is None else float(vmax)
    avg.plot(x="time", vmin=vmin, vmax=vmax)
    state.tape.save_plot()


commands = {"iterm": set_iterm_tape, "avg2d": avg2d, "hovmoller": hovmoller}


class ProgShell(cmd.Cmd):
    intro = (
        "Welcome to the ProgRunDiag shell.   Type help or ? to list commands.\n"  # noqa
    )

    def do_avg2d(self, arg):
        avg2d(loop_state, arg)

    def do_iterm(self, arg):
        set_iterm_tape(loop_state)

    def do_hovmoller(self, arg):
        hovmoller(loop_state, *arg.split())

    def do_load(self, arg):
        url = arg
        loop_state.load(url)

    def do_set(self, arg):
        key, val = arg.split()
        state[key] = val

    def do_print(self, arg):
        loop_state.print()

    def do_meridional(self, arg):
        variable = arg
        time = int(state.get("time", "0"))
        transect = meridional_transect(loop_state.data_3d.isel(time=time).merge(grid))
        transect[variable].plot(yincrease=False, y="pressure")
        loop_state.tape.save_plot()

    def onecmd(self, line):
        try:
            super().onecmd(line)
        except Exception as e:
            print(e)

    def do_map2d(self, arg):
        variable = arg
        time = int(state.get("time", "0"))
        data = loop_state.data_2d.isel(time=time)
        fv3viz.plot_cube(data, variable)
        time_name = data.time.item().isoformat()
        plt.title(f"{time_name} {variable}")
        plt.tight_layout()
        loop_state.tape.save_plot()

    def do_exit(self, arg):
        sys.exit(0)

    def do_eval(self, arg):
        f = pathlib.Path(arg)
        for line in f.read_text().splitlines():
            self.onecmd(line)


def register_parser(subparsers):
    parser = subparsers.add_parser("shell")
    parser.set_defaults(func=main)
    parser.add_argument("script", default="", nargs="?")


def main(args):
    shell = ProgShell()
    if args.script:
        shell.do_eval(args.script)
    else:
        shell.cmdloop()
