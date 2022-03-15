import report
import matplotlib.pyplot as plt


def test_MatplotlibFigure():
    plt.plot([0, 1], [0, 1])
    out = report.MatplotlibFigure(plt.gcf())
    out.source.startswith("<img")
    assert "src=" in out.source
