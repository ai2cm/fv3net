import subprocess
from itertools import product
from extractflow.utils import init_blob

timesteps = ['20160801.003000',
             '20160801.004500',
             '20160801.010000']
num_tiles = 1
num_subtiles = 1

prefix_src = ('gs://vcm-ml-data/'
              '2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/')
prefix_dst = 'gs://vcm-ml-data/tmp_dataflow/test_data_extract_check/'

src_template = (prefix_src +
                '{res}/{timestep}/'
                '{timestep}.{domain}.tile{tile:d}.nc.{subtile:04d}')

dst_template = (prefix_dst +
                '{res}/{timestep}/'
                '{timestep}.{domain}.tile{tile:d}.nc.{subtile:04d}')


items = product(timesteps, range(1, num_tiles+1), range(num_subtiles))
for tstep, tile, subtile in items:
    format_kwargs = dict(timestep=tstep, tile=tile, subtile=subtile)
    sfc_kwargs = dict(domain='sfc_data', res='C3702')

    # surface data
    src = src_template.format(**format_kwargs, **sfc_kwargs)
    dst = dst_template.format(**format_kwargs, **sfc_kwargs)

    subprocess.call(['gsutil', 'cp', src, dst])

    domains = ['fv_core_coarse.res', 'fv_srf_wnd_coarse.res',
               'fv_tracer_coarse.res']
    for domain in domains:
        domain_kwargs = dict(domain=domain, res='C384')
        src = src_template.format(**format_kwargs, **domain_kwargs)
        dst = dst_template.format(**format_kwargs, **domain_kwargs)
        subprocess.call(['gsutil', 'cp', src, dst])

    subprocess.call(['gsutil', 'cp',
                     prefix_src + f'C384/{tstep}/{tstep}.coupler.res',
                     prefix_dst + f'C384/{tstep}/{tstep}.coupler.res'])

# remove one from second timestep surface data
rm_sfc_file = dst_template.format(res='C3702', timestep=timesteps[1],
                                  domain='sfc_data', tile=1, subtile=0)
rm_atm_file = dst_template.format(res='C384', timestep=timesteps[2],
                                  domain='fv_srf_wnd_coarse.res',
                                  tile=1, subtile=0)

for rm_file in [rm_sfc_file, rm_atm_file]:
    subprocess.call(['gsutil', 'rm', rm_file])

# Create an empty timestep blob
bucket = 'vcm-ml-data'
empty_timestep = '20160801.011500'
res_list = ['C3702', 'C384']
blob_name = ('tmp_dataflow/test_data_extract_check/'
             '{res}/{timestep}/')
for res in res_list:
    curr_blob_name = blob_name.format(timestep=empty_timestep,
                                      res=res)
    blob = init_blob(bucket, curr_blob_name)
    blob.upload_from_string('')
