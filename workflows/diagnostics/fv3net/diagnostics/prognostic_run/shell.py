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


def meridional_transect(ds: xr.Dataset, lon):
    transect_coords = vcm.select.meridional_ring(lon)
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


class JupyterTape:
    def save_plot(self):
        pass


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
        self.state = {}

    def load(self, url):
        self.prognostic = load_run_data.SegmentedRun(url, catalog)
        self.data_3d = self.prognostic.data_3d.merge(grid)
        self.data_2d = grid.merge(self.prognostic.data_2d, compat="override")

    def get_time(self):
        i = int(self.state.get("time", "0"))
        return self.data_2d.time[i]

    def set(self, key, val):
        self.state[key] = val

    def get(self, key, default):
        return self.state.get(key, default)

    def get_3d_snapshot(self):
        time = self.get_time()
        return self.data_3d.sel(time=time, method="nearest").merge(grid)

    def get_2d_snapshot(self):
        time = self.get_time()
        return self.data_2d.sel(time=time)

    def print(self):
        print("3D Variables:")
        for v in self.data_3d:
            print(v)
        print()
        print("2D Variables:")
        for v in self.data_2d:
            print(v)

    def list_artifacts(self):
        for art in self.prognostic.artifacts:
            print(art)


catalog_path = vcm.catalog.catalog_path
catalog = intake.open_catalog(catalog_path)
grid = load_run_data.load_grid(catalog)


def avg2d(state: State, variable):
    x = state.data_2d
    avg = vcm.weighted_average(x[variable], x.area, ["x", "y", "tile"])
    avg.plot()
    state.tape.save_plot()


def avg3d(state: State, variable):
    x = state.data_3d
    avg = vcm.weighted_average(x[variable], x.area, ["x", "y", "tile"])
    avg.plot(y="pressure", yincrease=True)
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


def parse_pcolor_arg(arg):
    tokens = arg.split()
    kwargs = {}
    if len(tokens) >= 3:
        kwargs["vmin"] = float(tokens[1])
        kwargs["vmax"] = float(tokens[2])

    if len(tokens) >= 4:
        kwargs["cmap"] = tokens[3]

    return tokens[0], kwargs


class ProgShell(cmd.Cmd):
    intro = (
        "Welcome to the ProgRunDiag shell.   Type help or ? to list commands.\n"  # noqa
    )

    def __init__(self, state: State):
        super().__init__()
        self.state = state

    def do_avg2d(self, arg):
        avg2d(self.state, arg)

    def do_avg3d(self, arg):
        avg3d(self.state, arg)

    def do_iterm(self, arg):
        set_iterm_tape(self.state)

    def do_jupyter(self, arg):
        self.state.tape = JupyterTape()

    def do_hovmoller(self, arg):
        hovmoller(self.state, *arg.split())

    def do_artifacts(self, arg):
        self.state.list_artifacts()

    def do_load(self, arg):
        url = arg
        self.state.load(url)

    def do_set(self, arg):
        key, val = arg.split()
        self.state.set(key, val)

    def do_print(self, arg):
        self.state.print()

    def do_meridional(self, arg):
        variable, kwargs = parse_pcolor_arg(arg)
        lon = int(self.state.get("lon", "0"))
        transect = meridional_transect(self.get_3d_snapshot())
        transect = transect.assign_coords(lon=lon)
        transect[variable].plot(yincrease=False, y="pressure", **kwargs)
        self.state.tape.save_plot()

    def do_zonal(self, arg):
        variable, kwargs = parse_pcolor_arg(arg)
        lat = float(self.state.get("lat", 0))

        ds = self.state.get_3d_snapshot()

        transect_coords = vcm.select.zonal_ring(lat=lat)
        transect = vcm.interpolate_unstructured(ds, transect_coords)
        transect = transect.swap_dims({"sample": "lon"})
        transect = transect.assign_coords(lat=lat)

        plt.figure(figsize=(10, 3))
        transect[variable].plot(yincrease=False, y="pressure", **kwargs)
        self.state.tape.save_plot()

    def do_zonalavg(self, arg):
        variable, kwargs = parse_pcolor_arg(arg)
        ds = self.state.get_3d_snapshot()
        transect = vcm.zonal_average_approximate(ds.lat, ds[variable])
        transect.plot(yincrease=False, y="pressure", **kwargs)
        self.state.tape.save_plot()

    def do_column(self, arg):
        variable, kwargs = parse_pcolor_arg(arg)
        lon = float(self.state.get("lon", 0))
        lat = float(self.state.get("lat", 0))

        ds = self.state.get_3d_snapshot()
        transect_coords = vcm.select.latlon(lat, lon)
        transect = vcm.interpolate_unstructured(ds, transect_coords).squeeze()
        transect[variable].plot(yincrease=False, y="pressure", **kwargs)
        self.state.tape.save_plot()

    def onecmd(self, line):
        try:
            super().onecmd(line)
        except Exception as e:
            print(e)

    def do_map2d(self, arg):
        variable, kwargs = parse_pcolor_arg(arg)
        data = self.state.get_2d_snapshot()
        fv3viz.plot_cube(data, variable, **kwargs)
        time_name = data.time.item().isoformat()
        plt.title(f"{time_name} {variable}")
        plt.tight_layout()
        self.state.tape.save_plot()

    def do_exit(self, arg):
        sys.exit(0)

    def do_eval(self, arg):
        f = pathlib.Path(arg)
        for line in f.read_text().splitlines():
            self.onecmd(line)


def register_parser(subparsers):
    parser = subparsers.add_parser(
        "shell", help="Open an prognostic run browsing shell"
    )
    parser.set_defaults(func=main)
    parser.add_argument(
        "script",
        default="",
        nargs="?",
        help="If provided, a text file of commands to run instead of opening "
        "an interactive shell.",
    )


def main(args):
    shell = ProgShell(State())
    if args.script:
        shell.do_eval(args.script)
    else:
        shell.cmdloop()
