import kfp.dsl as dsl
from kfp import gcp

input_bucket="gs://vcm-ml-data/2019-11-06-X-SHiELD-gfsphysics-diagnostics-coarsened/C384"

def gcs_download_op(field):
    return dsl.ContainerOp(
        name='GCS - Download',
        image='google/cloud-sdk:216.0.0',
        command=['sh', '-c'],
        arguments=[f'mkdir $0; gsutil cp {input_bucket}/$0.tile?.nc $0/', field],
        file_outputs={
            'data': f'/{field}/',
        }
    ).apply(gcp.use_gcp_secret('user-sa'))


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


def list_op(tiles):
    return dsl.ContainerOp(name=f'ls', image='google/cloud-sdk',
                           command=['sh', '-c'],
                           arguments=['ls', tiles])



@dsl.pipeline(
    name='Regrid the input data',
)
def download_save_most_frequent_word():
    input_data = [
        'DLWRFsfc', 'DSWRFsfc', 'DSWRFtoa', 'HPBLsfc', 'LHTFLsfc', 'PRATEsfc', 'SHTFLsfc',
        'UGRD10m', 'ULWRFsfc', 'ULWRFtoa', 'USWRFsfc', 'USWRFtoa', 'VGRD10m', 'uflx', 'vflx'
    ][0:1]

    for field in input_data:
        tiles = gcs_download_op(field)
        list_op(tiles.outputs['data'])
        # downloader = RegridOp(field).apply(gcp.use_gcp_secret('user-gcp-sa'))


