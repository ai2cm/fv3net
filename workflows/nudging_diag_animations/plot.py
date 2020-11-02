from utils import global_average
import fv3viz as viz
from matplotlib import pyplot as plt, animation, rc
import cartopy.crs as ccrs

rc("animation", html="html5")

MAPPABLE_VAR_KWARGS = {
    "coord_x_center": "x",
    "coord_y_center": "y",
    "coord_x_outer": "x_interface",
    "coord_y_outer": "y_interface",
    "coord_vars": {
        "lonb": ["y_interface", "x_interface", "tile"],
        "latb": ["y_interface", "x_interface", "tile"],
        "lon": ["y", "x", "tile"],
        "lat": ["y", "x", "tile"],
    },
}


def plot_comparison_frame(
    time, ds, var, axes, plot_cube_kwargs_abs, plot_cube_kwargs_diff
):
    time_str = time.item().strftime("%Y%m%d.%H%M%S")
    for ax, derivation in zip(axes[:2], ds.derivation[:2]):
        ax.clear()
        colorbar = True if derivation.item().startswith("C48") else False
        viz.plot_cube(
            viz.mappable_var(
                ds.sel(time=time, derivation=derivation.item()),
                var,
                **MAPPABLE_VAR_KWARGS,
            ),
            ax=ax,
            colorbar=colorbar,
            **plot_cube_kwargs_abs,
        )
        ax.set_title(f"{time_str}: {derivation.item()}")
    axes[2].clear()
    viz.plot_cube(
        viz.mappable_var(
            ds.sel(time=time, derivation="difference"), var, **MAPPABLE_VAR_KWARGS
        ),
        ax=axes[2],
        **plot_cube_kwargs_diff,
    )
    axes[2].set_title(f"{time_str}: reference minus C48")


def make_comparison_animation(
    ds,
    var,
    plot_cube_kwargs_abs=None,
    plot_cube_kwargs_diff=None,
    fig_size=[12, 7],
    dpi=100,
    **func_animation_kwargs,
):
    plot_cube_kwargs_abs = plot_cube_kwargs_abs or {}
    plot_cube_kwargs_diff = plot_cube_kwargs_diff or {}
    fig = plt.figure()
    axes = list(
        fig.subplots(2, 2, subplot_kw={"projection": ccrs.Robinson()}).flatten()
    )
    axes[-1].remove()
    axes = axes[:3]
    axll = fig.add_subplot(2, 2, 4)
    axes.append(axll)
    h = ds[f"{var}_land_average"].plot(ax=axll, hue="derivation", add_legend=False)
    axll.legend(h, ds.derivation.values)
    axll.yaxis.set_label_position("right")
    axll.yaxis.tick_right()
    axll.set_ylabel(
        f'{ds[var].attrs.get("long_name", var)} [{ds[var].attrs.get("units", "-")}]'
    )
    axll.set_title(f"{var}: land average")
    axll.set_xlim([ds.time[0].item(), ds.time[-1].item()])
    anim = animation.FuncAnimation(
        fig,
        plot_comparison_frame,
        frames=ds.time,
        fargs=(ds, var, axes, plot_cube_kwargs_abs, plot_cube_kwargs_diff),
        **func_animation_kwargs,
    )
    fig.set_size_inches(fig_size)
    fig.set_dpi(dpi)
    plt.close(fig)
    return anim


def plot_nudging_frame(time, ds, var, ax, plot_cube_kwargs):
    time_str = time.item().strftime("%Y%m%d.%H%M%S")
    ax.clear()
    viz.plot_cube(
        viz.mappable_var(ds.sel(time=time).isel(z=-1), var, **MAPPABLE_VAR_KWARGS),
        ax=ax,
        **plot_cube_kwargs,
    )
    ax.set_title(f"{time_str}: lowest level {var}")


def make_nudging_animation(
    ds, var, plot_cube_kwargs=None, fig_size=[12, 5], dpi=100, **func_animation_kwargs
):
    plot_cube_kwargs = plot_cube_kwargs or {}
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax2 = fig.add_subplot(1, 2, 2)
    global_average(ds[var], ds["area"], ds["SLMSK"], "land").plot(
        x="time",
        y="z",
        ax=ax2,
        cbar_kwargs={"label": ds[var].attrs["units"], "pad": 0.2},
    )
    ax2.invert_yaxis()
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel("model level")
    ax2.set_title(f"{var}: land average")
    ax2.set_xlim([ds.time[0].item(), ds.time[-1].item()])
    anim = animation.FuncAnimation(
        fig,
        plot_nudging_frame,
        frames=ds.time,
        fargs=(ds, var, ax1, plot_cube_kwargs),
        **func_animation_kwargs,
    )
    fig.set_size_inches(fig_size)
    fig.set_dpi(dpi)
    plt.close(fig)
    return anim
