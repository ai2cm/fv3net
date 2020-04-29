from report import create_html, HTMLPlot
from generate_report import save
import holoviews as hv


def test_hv_to_html():
    p = hv.Curve([(0,1), (1,0)])
    html = save(p)
    assert "</html>" not in html


def test_rpeort():
    m = hv.HoloMap()
    p = hv.Curve([(0,1), (1,0)])
    m['a'] = p
    p = hv.Curve([(0,2), (2,0)])
    m['b'] = p
    head = """
     <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-2.0.2.min.js" integrity="sha384-ufR9RFnRs6lniiaFvtJziE0YeidtAgBRH6ux2oUItHw5WTvE1zuk9uzhUU/FJXDp" crossorigin="anonymous"></script>
     <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.0.2.min.js" integrity="sha384-8QM/PGWBT+IssZuRcDcjzwIh1mkOmJSoNMmyYDZbCfXJg3Ap1lEvdVgFuSAwhb/J" crossorigin="anonymous"></script>
    <script type="text/javascript" src="https://unpkg.com/@holoviz/panel@^0.9.5/dist/panel.min.js" integrity="sha384-" crossorigin="anonymous"></script>
    """
    div = save(m)

    html = create_html({"main": [HTMLPlot(div)]}, title="Hello", html_header=head)
    print(html)
    hv.save(m, "out.html")
    with open("index.html", "w") as f:
        f.write(html)


