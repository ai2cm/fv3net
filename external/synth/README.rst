Integration tests with synthetatic data

1. Generate schema of big zarr data with:

    python zarr_to_test_schema.py > schema.json

1. Run integration tests :
   
    pytest test_integration.py

Saves output to disk
