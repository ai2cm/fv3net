set -e
set -x
kustomizations=$(find . -name 'kustomization.yaml')

for k in $kustomizations; do
	echo "Validating $k"
	kubectl kustomize $(dirname $k)
done
