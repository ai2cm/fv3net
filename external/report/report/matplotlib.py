import base64
import io

from .create_report import RawHTML


def fig_to_b64(fig, format="png", dpi=None):
    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format=format, bbox_inches="tight", dpi=dpi)
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())
    return f"data:image/png;base64, " + pic_hash.decode()


def MatplotlibFigure(fig, width=None) -> RawHTML:
    """Convert a matplotlib figure into a image tag."""
    properties = {}
    if width:
        properties["width"] = width
    properties["src"] = fig_to_b64(fig)
    properties_str = " ".join(f'{key}="{val}"' for key, val in properties.items())
    return RawHTML("<img " + properties_str + "/>")
