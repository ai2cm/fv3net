usage="Usage: run.sh <url> [gcs]"

if [[ $# < 1 ]]
then
	echo $usage
fi


url=$1
(
	export PROG_RUN_LOCATION=$1
	jupyter nbconvert --execute prognostic-run-diags-v1.ipynb
)

if [[ $# > 1]]
then
	gsutil cp prognostic-run-diags-v1.html $2
fi
