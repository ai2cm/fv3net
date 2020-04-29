from report.holoviews import HVPlot


def test_HVPlot():
    import holoviews as hv
    hv.extension("bokeh")
    m = hv.HoloMap()
    p = hv.Curve([(0, 1), (1, 0)])
    m["a"] = p
    p = hv.Curve([(0, 2), (2, 0)])
    m["b"] = p

    plot = HVPlot(m)
    out = plot.render()
    assert isinstance(out, str)

    # ensure that this output only contains a div and script
    # not a complete html page
    assert "<html/>" not in out
