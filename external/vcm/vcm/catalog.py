import os
import intake

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
catalog_path = os.path.join(FILE_DIR, "catalog.yml")
catalog = intake.open_catalog(catalog_path)

