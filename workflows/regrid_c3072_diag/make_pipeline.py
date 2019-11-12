import kfp.dsl as dsl
from kfp import gcp


def get_regrid_command(field):
   return ['/usr/bin/regrid.sh', field]


class RegridOp(dsl.ContainerOp):
    """Regrid a set of netcdf files using fregrid
    """

    def __init__(self, field):
        """Args:
             message: a dsl.PipelineParam object representing an input message.
        """
        super(RegridOp , self).__init__(
            name='regrid-'+ field,
            image='us.gcr.io/vcm-ml/regrid_c3072_diag',
            command=get_regrid_command(field))


@dsl.pipeline(
    name='Regrid the input data',
)
def download_save_most_frequent_word():
    input_data = [
        'DLWRFsfc', 'DSWRFsfc', 'DSWRFtoa', 'HPBLsfc', 'LHTFLsfc', 'PRATEsfc', 'SHTFLsfc',
        'UGRD10m', 'ULWRFsfc', 'ULWRFtoa', 'USWRFsfc', 'USWRFtoa', 'VGRD10m', 'uflx', 'vflx'
    ]

    for field in input_data:
        RegridOp(field).apply(gcp.use_gcp_secret('user-gcp-sa'))


