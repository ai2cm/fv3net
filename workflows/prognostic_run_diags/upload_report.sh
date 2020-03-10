set -x
jupyter nbconvert --execute combined.ipynb
python putfile.py combined.html gs://vcm-ml-public/testing-2020-02/prognostic_run_diags/combined.html