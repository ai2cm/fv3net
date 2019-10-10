"""
work_dir=/mnt/disks/fat-disk/work_dir

mkdir -p $work_dir
echo "Changing directory to $work_dir"
cd $work_dir

file=fv_srf_wnd_coarse.res.tile1.tar
path=gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-restart-tarballs/$file

files=fv_srf_wnd_coarse.res.tile1.tar
gsBaseDir=gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-restart-tarballs
outputBaseDir=gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-restart-extracted/


files=$(gsutil ls $gsBaseDir/*.tar)
echo Will process these files: 
echo $files
read -p "Process these files" key

case $key in 
	n)
		echo "Aborting"
		exit 1
		;;
	*)
		echo "continuing"
		;;
esac



for remoteFile in $files
do
	file=$(basename $remoteFile)
	echo "Processing $file"
	fileList=extracted_files.${file}.txt 
	[ -f $file ] || gsutil cp $gsBaseDir/$file .
	tar xvf $file | tee $fileList |  gsutil -m cp -I $outputBaseDir > gsutil.transfer.${file}.log 2>&1
	cat $fileList | xargs rm -r
	rm $file
done
"""
from prefect import task, Flow
from datetime import timedelta

from glob import glob
from os.path import join, basename
import os
from subprocess import check_call
from prefect.tasks.shell import ShellTask
from toolz import groupby

from prefect.engine.executors import DaskExecutor

executor = DaskExecutor(local_processes=True)

bucket="gs://vcm-ml-data/2019-10-03-X-SHiELD-C3072-to-C384-restart-tarballs"
file="fv_srf_wnd_coarse.res.tile1.tar"
output="work_dir/fv_srf_wnd_coarse.res.tile1"
output_bucket ="gs://vcm-ml-data/2019-10-07-testing"

shell = ShellTask(helper_script="cd ~/fat-disk/work_dir", max_retries=10, retry_delay=timedelta(seconds=10))

@task(max_retries=10, retry_delay=timedelta(seconds=10))
def tar(group):
    tag, files = group
    tarfile = basename(f'{tag}.tar')
    check_call(['tar', 'cf', tarfile] + files)
    upload(tarfile)
    os.unlink(tarfile)
    return tarfile


@task
def group_subtiles(ncfiles):
    out = list(groupby(lambda x: x[:-5], ncfiles).items())
    return out

def upload(tarfile):
    check_call([ 'gsutil', 'cp', tarfile, join(output, basename(tarfile))])
    return tarfile

@task
def print_t(f):
    print(f)

@task
def listnc(dir):
    path = join(dir, '*.nc.????')
    return glob(path)


def download_data_command(file):
    url = join(bucket, file)
    return f"gsutil -o 'GSUtil:parallel_thread_count=1' -o 'GSUtil:sliced_object_download_max_component=32' cp  {url} ."


with Flow("Download and tar") as f:
    download = shell(command=download_data_command(file))
    extract = shell(command=f"tar xvf {file}")
    extract.set_upstream(download)
    ncfiles = listnc(output, upstream_tasks=[extract])
    groups = group_subtiles(ncfiles)
    tarfiles = tar.map(groups)
    cleanup = task(lambda : os.unlink(join('work_dir', file))
    cleanup.set_upstream(tarfiles)


#f.run()
f.run(executor=executor)
