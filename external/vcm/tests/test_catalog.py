import intake
import vcm.catalog


def test_catalog():
    assert isinstance(vcm.catalog.catalog, intake.Catalog)
