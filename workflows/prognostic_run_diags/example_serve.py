from report import create_html, HTMLPlot
from bokeh.embed import components
import holoviews as hv
from bokeh.embed import json_item, file_html
from bokeh.resources import CDN
import json
import panel
from bokeh.resources import CDN
hv.extension('bokeh')


template = """
<html>
<head>
        <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-2.0.2.min.js" integrity="sha384-ufR9RFnRs6lniiaFvtJziE0YeidtAgBRH6ux2oUItHw5WTvE1zuk9uzhUU/FJXDp" crossorigin="anonymous"></script>
        <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.0.2.min.js" integrity="sha384-8QM/PGWBT+IssZuRcDcjzwIh1mkOmJSoNMmyYDZbCfXJg3Ap1lEvdVgFuSAwhb/J" crossorigin="anonymous"></script>
        <script type="text/javascript" src="https://unpkg.com/@holoviz/panel@^0.9.5/dist/panel.min.js" integrity="sha384-" crossorigin="anonymous"></script>
</head>
<body>
<div id="myplot" />
<script>
var item = {}
Bokeh.embed.embed_item(item)
</script>
</body>
</html>
"""

m = hv.HoloMap()
p = hv.Curve([(0,1), (1,0)])
m['a'] = p
p = hv.Curve([(0,2), (2,0)])
m['b'] = p



# save with holoviews
# interaction works
hv.save(m, "out.html")

# save with bokeh embedding + template
# interaction doesn't work
pane = panel.panel(m)
# save all holomaps
pane.embed()
model = pane.get_root()
s = json.dumps(json_item(model, 'myplot'))
out = template.format(s)
with open('index.html', 'w') as f:
    f.write(out)

# save with bokeh file_html
# interaction doesn't work
with open('index_file_html.html', 'w') as f:
    f.write(file_html(model, resources=CDN))


# save with panel
# interaction doesn't work
pane.save("index_with_panel.html", resources=CDN)

# renderer components
# works
template = """
<html>
<head>
</head>
<body>
{}
</body>
</html>
"""
r = hv.renderer('bokeh')
html, js = r.components(m)
with open("index_hv_rendered_components.html", "w") as f:
    f.write(template.format(html['text/html']))