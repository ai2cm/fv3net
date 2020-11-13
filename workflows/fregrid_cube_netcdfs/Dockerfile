FROM us.gcr.io/vcm-ml/fretools

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install xarray netCDF4

COPY regrid.sh /usr/bin/regrid.sh
RUN chmod +x /usr/bin/regrid.sh
COPY prognostic_diags_to_tiled.py /usr/bin/prognostic_diags_to_tiled.py
COPY print_fields.py /usr/bin/print_fields.py
COPY regrid_prognostic_run_diags.sh /usr/bin/regrid_prognostic_run_diags.sh
RUN chmod +x /usr/bin/regrid_prognostic_run_diags.sh
ENTRYPOINT /usr/bin/regrid.sh
