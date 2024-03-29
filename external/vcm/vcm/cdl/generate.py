from typing import Mapping
import lark
import numpy as np
import xarray

from .parser import grammar


def parse_value_node(value_node):
    if value_node.data == "num":
        val = float(value_node.children[0].value)
    elif value_node.data == "nan":
        val = np.nan
    elif value_node.data == "string":
        val = value_node.children[0].value[1:-1]
    else:
        raise ValueError(f"Cannot parse {value_node.data}")
    return val


class DatasetVisitor(lark.Visitor):
    """A class for traversing a parsed syntax tree

    This class will iterate over the syntax tree of a CDL file and generate a
    list of variable information that can be used to generate an xarray Dataset.

    For example, the ``.variable_decl`` method will be called when the tree
    'visits' a node of type ``variable_decl``. Note how these method names match
    the rules in the vcm.cdl.grammar.
    """

    def __init__(self):
        self._variable_dims = {}
        self._variable_dtype = {}
        self._variable_attrs = {}
        self._dims: Mapping[str, int] = {}
        self._variable_data = {}

    def dimension_pair(self, v):
        dim = v.children[0].value
        size = (
            int(v.children[1].value)
            if v.children[1].type == "INT"
            else v.children[1].value
        )
        self._dims[dim] = size

    def variable_decl(self, v):
        dtype_node, name, dims_node = v.children
        name = str(name)
        dtype_str = dtype_node.data
        dims = [str(dim) for dim in dims_node.children] if dims_node else []
        self._variable_dims[name] = dims
        self._variable_dtype[name] = dtype_str

    def variable_attr(self, v):
        varname, attr_node, value_node = v.children
        attrname = attr_node.value
        attrs = self._variable_attrs.setdefault(varname.value, {})
        attrs[attrname] = parse_value_node(value_node)

    def datum(self, v):
        varname = v.children[0].value
        assert varname in self._variable_dims
        # get data stored in self.list
        self._variable_data[varname] = list(v.children[1].data)

    def list(self, v):
        # Store the data in the tree to be used later on in self.datum
        v.data = [parse_value_node(node) for node in v.children]

    def generate_dataset(self):
        data_vars = {}
        for name in self._variable_dims:
            shape = tuple(self._dims[k] for k in self._variable_dims[name])
            arr = np.zeros(shape)
            if name in self._variable_data:
                data = self._variable_data[name]
                view = arr.ravel()
                n = min(view.size, len(data))
                view[:n] = data[:n]

            data_vars[name] = (
                self._variable_dims[name],
                arr,
                self._variable_attrs.get(name, {}),
            )
        return xarray.Dataset(data_vars)


def cdl_to_dataset(cdl: str) -> xarray.Dataset:
    """Convert a CDL string into a xarray dataset

    Useful for generating synthetic data for testing

    The UCAR Common Data Language(`CDL`_) is a human readable format with the
    same data model as netCDF.  CDL can be translated to binary netCDF using the
    `ncgen` command line tool bundled with netCDF. CDL is very compact and looks
    like this::

        netcdf example {   // example of CDL notation
        dimensions:
            lon = 3 ;
            lat = 8 ;
        variables:
            float rh(lon, lat) ;
                rh:units = "percent" ;
                rh:long_name = "Relative humidity" ;
        // global attributes
            :title = "Simple example, lacks some conventions" ;
        data:
        rh =
            2, 3, 5, 7, 11, 13, 17, 19,
            23, 29, 31, 37, 41, 43, 47, 53,
            59, 61, 67, 71, 73, 79, 83, 89 ;
        }

    .. _CDL: https://www.unidata.ucar.edu/software/netcdf/workshops/most-recent/nc3model/Cdl.html

    """  # noqa
    # create the parser
    cdl_parser = lark.Lark(grammar, start="netcdf", parser="lalr")
    # parse the string into a syntax tree
    tree = cdl_parser.parse(cdl)
    # collects lists of all the variables, data, and metadata
    v = DatasetVisitor()
    v.visit(tree)
    # finally generate the dataset
    ds = v.generate_dataset()
    return xarray.decode_cf(ds)
