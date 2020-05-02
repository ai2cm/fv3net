import holoviews as hv


class HVPlot:
    """Renders holoviews plots to HTML for use in the diagnostic reports
    """

    def __init__(self, hvplot):
        self._plot = hvplot

    def __repr__(self) -> str:
        # It took hours to find this combinitation of commands!
        # it was really hard finding a combintation that
        # 1. embedded the data for an entire HoloMap object
        # 2. exported the html as a div which can easily be embedded in the reports.
        r = hv.renderer("bokeh")
        html, _ = r.components(self._plot)
        html = html["text/html"]
        return html
