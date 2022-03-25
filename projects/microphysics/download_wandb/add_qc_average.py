# flake8: noqa
# %%
import sqlite3
from fv3net.diagnostics.prognostic_run.emulation import single_run
import xarray

query = """
select id, group_, json_extract(config, '$.rundir.value') as url
from runs
where group_ in (
  'cycle-trained-limited-eb2271-v1-offline'
  ,'iter2-cycle-v2-online'
  ,'cycle-trained-limited-eb2271-v1-online'
  ,'online-12hr-cycle-v3-online'
  ,'online-12hr-cycle-v3-offline'
) AND job_type='prognostic_run'
order by created_at;
"""

drop_table_sql = "DROP TABLE IF EXISTS average;"
create_table_sql = """
CREATE TABLE average (
    run_id text,
    time text,
    cloud real
);
"""


def create_table():
    cur = con.cursor()
    cur.execute(drop_table_sql)
    cur.execute(create_table_sql)
    cur.close()
    con.commit()


con = sqlite3.connect("example.db")

# %%
create_table()

import vcm

# %%


def compute_qc_global(id, url):
    print(url)
    import fsspec

    mapper = fsspec.get_mapper(url + "/atmos_dt_atmos.zarr")
    try:
        ds = xarray.open_zarr(mapper)
    except:
        print(f"skipping {url}")
        return
    cloud = vcm.weighted_average(
        ds.VIL, ds.area, dims=["tile", "grid_xt", "grid_yt"]
    ).load()
    cur = con.cursor()
    for time in cloud.time.values:
        cur.execute(
            "INSERT INTO average (run_id, time, cloud) VALUES (?,?,?)",
            (id, vcm.cast_to_datetime(time).isoformat(), float(cloud.sel(time=time))),
        )
    cur.close()
    con.commit()


# %%
cur = con.cursor()
for row in cur.execute(query):
    id, group_, url = row
    compute_qc_global(id, url)


# %%
con.close()

# %%
