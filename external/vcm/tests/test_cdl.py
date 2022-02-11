import numpy as np
import lark
import vcm
from vcm.cdl.parser import grammar as parser
from vcm.cdl.generate import DatasetVisitor


def test_get_data():

    ds = vcm.cdl_to_dataset(
        """
    netcdf Some Data {
    dimensions:
        time = 3;
        x = 4;
    variables:
        int time(time);
        int b;
        double a(time, x);
            a:_FillValue = 0;
            a:foo = "bar";

    data:
        time = 1,2,3;
        b = 3;
    }
    """
    )

    assert ds["a"].dims == ("time", "x")
    assert np.all(np.isnan(ds["a"]))
    assert ds["time"].values.tolist() == [1, 2, 3]
    assert ds.a.foo == "bar"
    assert ds.b.item() == 3


def test_lark():

    cdl_parser = lark.Lark(parser, start="dimensions")
    print(cdl_parser.parse("dimensions: a = 1; b=3;"))

    cdl_parser = lark.Lark(parser, start="variables")
    print(cdl_parser.parse("variables: float a(x,y); int b(y); int c;"))

    cdl_parser = lark.Lark(parser, start="variables")
    print(cdl_parser.parse("variables: float a(x,y); a:someAttr = 0; int b(y);"))

    cdl_parser = lark.Lark(parser, start="value")
    print(cdl_parser.parse("NaN"))

    cdl_parser = lark.Lark(parser, start="datum")
    print(cdl_parser.parse("time =  1, 2, 3;"))

    cdl_parser = lark.Lark(parser, start="netcdf")
    tree = cdl_parser.parse(
        """
    netcdf Some Data {
    dimensions:
        time = 3;
        x = 4;
    variables:
        int time(time);
        double a(time, x);
            a:_FillValue = 0;

    data:
        time = 1,2,3;
    }
    """
    )
    print(tree)

    tree = cdl_parser.parse(
        """
    netcdf Some Data {
    dimensions:
        time = 3;
        x = 4;
    variables:
        int time(time);
        double a(time, x);
            a:_FillValue = 0;

    data:
        time = 1,2,3;
        x = 1,2,3,4;
    group: SubGroup {
        dimensions:
            time = 1;
        variables:
            int time(time);
    }
    }
    """
    )
    print(tree.pretty())


def test_parse_data_value():
    cdl_parser = lark.Lark(parser, start="variable_decl")
    tree = cdl_parser.parse("float a(x,y);")
    v = DatasetVisitor()
    v.visit(tree)
    assert v._variable_dtype["a"] == "float"
    assert v._variable_dims["a"] == ["x", "y"]


def test_Visitor():
    cdl_parser = lark.Lark(parser, start="netcdf")
    cdl = """netcdf Some Data {
    dimensions:
        time = 3;
        x = 4;
    variables:
        int time(time);
        double a(time, x);
            a:_FillValue = 0;
            time:somefun = NaN;

    data:
        time = 1,2,3;
    }"""
    tree = cdl_parser.parse(cdl)
    v = DatasetVisitor()
    v.visit(tree)
    assert v._dims == {"time": 3, "x": 4}

    assert v._variable_attrs["a"] == {"_FillValue": 0}
    assert np.isnan(v._variable_attrs["time"]["somefun"])
