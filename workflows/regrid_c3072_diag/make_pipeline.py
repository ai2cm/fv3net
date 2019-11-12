import kfp.dsl as dsl
from kfp import gcp


def get_regrid_command(source_prefix, output_bucket, resolution, fields: str):
    return ['/usr/bin/regrid.sh', source_prefix, output_bucket, resolution,
            fields]


def regrid_op(*args, **kwargs):
    return dsl.ContainerOp(
        name='regrid',
        image='us.gcr.io/vcm-ml/regrid_c3072_diag',
        command=get_regrid_command(*args, **kwargs),
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


@dsl.pipeline(
    name='Regrid the input data', )
def regrid_input_data(source_prefix: str, output_bucket: str, resolution: str,
                      fields: str):
    regrid_op(source_prefix, output_bucket, resolution, fields)
