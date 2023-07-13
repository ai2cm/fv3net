import cdsapi
c = cdsapi.Client()
MONTHS = ["07", "08", "09", "10", "11", "12"] 
for month in MONTHS:
  result = c.service(
   "tool.toolbox.orchestrator.workflow",
  params={
   "realm": "user-apps",
   "project": "app-c3s-daily-era5-statistics",
   "version": "master",
   "kwargs": {
   "dataset": "reanalysis-era5-single-levels",
   "product_type": "reanalysis",
   "variable": "sea_surface_temperature",
   "statistic": "daily_mean",
   "year": "1955",
   "month": month,
   "time_zone": "UTC+00:00",
   "frequency": "1-hourly",
   "grid": "1.0/1.0",
   "area": {"lat": [-90, 90], "lon": [-180, 180]}
   },
   "workflow_name": "application"
   })
  c.download(result) 
