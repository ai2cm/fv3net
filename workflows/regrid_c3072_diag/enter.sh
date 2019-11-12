
bindMounts="-v /home/noahb:/home/noahb -w $(pwd) -ti "
docker run $bindMounts us.gcr.io/vcm-ml/fretools  bash
