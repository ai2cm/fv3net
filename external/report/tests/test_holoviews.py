from report.holoviews import HVPlot, get_html_header


def test_HVPlot():
    import holoviews as hv

    hv.extension("bokeh")
    m = hv.HoloMap()
    p = hv.Curve([(0, 1), (1, 0)])
    m["a"] = p
    p = hv.Curve([(0, 2), (2, 0)])
    m["b"] = p

    plot = HVPlot(m)
    out = repr(plot)
    assert isinstance(out, str)

    # ensure that this output only contains a div and script
    # not a complete html page
    assert "<html/>" not in out


def test_get_html_header():
    html = get_html_header()

    assert "<html" not in html
    assert "<script" in html
    # self closing script not allowed in html
    assert " />" not in html
