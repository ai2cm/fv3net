import kfp.dsl as dsl
from kfp import gcp


def get_regrid_command(source_prefix, output_bucket, resolution, fields: str, extra_args):
    return ['/usr/bin/regrid.sh', source_prefix, output_bucket, resolution,
            fields, extra_args] 


def regrid_op(*args, **kwargs):
    return dsl.ContainerOp(
        name='regrid',
        image='us.gcr.io/vcm-ml/regrid_c3072_diag',
        command=get_regrid_command(*args, **kwargs),
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


@dsl.pipeline(
    name='Regrid the input data', )
def regrid_input_data(source_prefix: str="gs://vcm-ml-data/2019-11-12-fv3gfs-C48-five-day-nudged-run/nudge_rundir/sfc_dt_atmos",
        output_bucket: str="gs://vcm-ml-data/2019-11-12-fv3gfs-C48-five-day-nudged-run/nLat180_nLon360/sfc_dt_atmos.nc", 
        resolution: str="C48",
        fields: str="DLWRFsfc,DSWRFsfc,DSWRFtoa,HPBLsfc,LHTFLsfc,PRATEsfc,SHTFLsfc,ULWRFsfc,ULWRFtoa,USWRFsfc,USWRFtoa",
        extra_args: str='--nlat 180 --nlon 360'):
    regrid_op(source_prefix, output_bucket, resolution, fields, extra_args)
