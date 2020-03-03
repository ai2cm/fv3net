usage="Usage: run.sh <url>"

if [[ $# < 1 ]]
then
	echo $usage
fi

url=$1
(
	export PROG_RUN_LOCATION=$1
	jupyter nbconvert --execute prognostic-run-diags-v1.ipynb
)
