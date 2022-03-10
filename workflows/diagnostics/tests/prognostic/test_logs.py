from datetime import timedelta
from fv3net.diagnostics.prognostic_run.logs import parse_duration


def test_parse_duration():
    logs = [
        """
INFO:profiles:{"time": "2016-06-20T00:15:00"}
DEBUG:runtime.loop:Dynamics Step
DEBUG:runtime.loop:Applying prephysics state updates for: []
DEBUG:runtime.loop:Physics Step (compute)
INFO:profiles:{"time": "2016-06-20T00:30:00"}
INFO:profiles:{"time": "2016-06-20T00:45:00"}
DEBUG:runtime.loop:Dynamics Step
DEBUG:runtime.loop:Applying prephysics state updates for: []
        """
    ]
    assert timedelta(minutes=45) == parse_duration(logs)
