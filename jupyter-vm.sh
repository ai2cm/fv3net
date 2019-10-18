#!/bin/bash

# NAME
#    jupyter_vm.sh
# SYNOPSIS
#    jupyter_vm <cmd> <instance_name> <env_name> where <cmd> is 'start', 'reconnect', 'stop', or 'ssh-connect', or 'port-forward', <instance_name>
#    is a pre-existing compute instance, and <env_name> is a conda environment to use on the compute instance (if desired, otherwise leave blank)
#     - 'start': i.e., jupyter_vm start <instance_name> <env_name> 
#        starts a jupyter lab instance on the remote machine, configures port-forwarding via ssh, and launches the server url in a local browser
#     - 'reconnect', i.e., jupyter_vm reconnect <instance_name> <env_name>
#        reestablishes the port-forwarding to a running remote jupyter instance, e.g., to be used after ssh disconnection
#     - 'stop', i.e., jupyter_vm stop <instance_name>
#        closes all jupyter instances on the remote machine and stops the instance
#     - 'ssh-connect', i.e., jupyter_vm ssh-connect <instance_name>
#        connects to the remote machine directly with ssh
#     - 'port-forward', i.e., jupyter_vm port-forward <instance_name> <port_number> 
#        establishes port-forwarding on the port specified by <port_number>; not needed if the remote jupyter instance was started using 'start'
#        but useful if it was started in another manner, or for port-forwarding remote docker containers
# NOTES
#    assumes MacOS local machine
#    requires an existing gcloud compute instance set up with conda, jupyter lab, and tmux
#    requires local gcloud account credentials to be set (gcloud auth login)
#    requires the following macos packages:
#         -gnu grep, i.e., brew install grep
#    also requires that the following environment variables be set on the user's local machine:
#         -GCLOUD_COMPUTE_USER: username on remote server
#         -GCLOUD_COMPUTE_KEYFILE: ssh key file location on local machine

# Brian Henn, Vulcan Climate Modeling, September 2019

CMD=$1
INSTANCE_NAME=$2
ENV_NAME=$3
USER=$GCLOUD_COMPUTE_USER 
KEYFILE=$GCLOUD_COMPUTE_KEYFILE

if [ "$1" == 'start' ]; then
    
    # kill open ports left by ssh tunnels first
    PIDS=$(lsof -i -P -n | grep 'ssh.*LISTEN' | awk '{print $2}' | uniq)
    for pid in $PIDS; do
        kill $pid
    done

    # start the instance if needed using the name to get the ID
    INSTANCE_STATUS=$(gcloud compute instances list --filter="name~""${INSTANCE_NAME}" | grep -e "${INSTANCE_NAME}" | grep -o '\d*\.\d*\.\d*\.\d*.*' | awk '{print $NF}')
    if [ ! $INSTANCE_STATUS == 'RUNNING' ]; then
       echo "Running command \"gcloud compute instances start "${INSTANCE_NAME}"\"" 
       gcloud compute instances start ${INSTANCE_NAME} > /dev/null
       # let it spin up
       echo 'Spinning up gcloud compute instance...'
       sleep 10s
    fi

    # get the new host address once it's spun up
    HOST=$(gcloud compute instances list --filter="name~${INSTANCE_NAME}" | grep -e ${INSTANCE_NAME} | grep -o '\d*\.\d*\.\d*\.\d*' | tail -n1)

    # check to see if there are already instances running on the server
    servers=$(ssh -i "${KEYFILE}" -o "StrictHostKeyChecking no" ${USER}@${HOST} 'eval "$('\${HOME}/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"; conda activate '${ENV_NAME}'; jupyter notebook list')
    if [[ $servers == *http* ]]; then
        echo "At least one jupyter server already running, try 'reconnect' option."
        exit
    fi

    # ssh to the instance, start the jupyter-lab server, and get the url
    echo "Running command \"ssh -i "${KEYFILE}" -o \"StrictHostKeyChecking no\" -fL 9999:localhost:9999 "${USER}"@"${HOST} \
        "eval \"\$('\${HOME}/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)\"; conda activate "${ENV_NAME}"; tmux; jupyter-lab --no-browser --port=9999 --notebook-dir='/'\"" 
    output=$(ssh -i "${KEYFILE}" -o "StrictHostKeyChecking no" -fL 9999:localhost:9999 ${USER}@${HOST} \
        'eval "$('/home/brianh/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"; conda activate '${ENV_NAME}'; tmux 2> /dev/null; jupyter-lab --no-browser --port=9999 --notebook-dir=\"/\"' > tmp.txt 2>&1)
    sleep 10s
    url=$(cat tmp.txt | grep -o 'http://localhost\S*' | tail -n1)
    echo "Opening ""$url"" in browser window."
    open $url
    rm -f tmp.txt
   
elif [ "$1" == 'reconnect' ]; then
   
    # kill open ports left by ssh tunnels first
    PIDS=$(lsof -i -P -n | grep 'ssh.*LISTEN' | awk '{print $2}' | uniq)
    for pid in $PIDS; do
        kill $pid
    done   

    # get the new host address
    HOST=$(gcloud compute instances list --filter="name~${INSTANCE_NAME}" | grep -e ${INSTANCE_NAME} | grep -o '\d*\.\d*\.\d*\.\d*' | tail -n1)
   
    # check to see if there are already instances running on the server
    servers=$(ssh -i "${KEYFILE}" -o "StrictHostKeyChecking no" ${USER}@${HOST} 'eval "$('\${HOME}/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"; conda activate '${ENV_NAME}'; jupyter notebook list')
    if [[ ! $servers == *http* ]]; then
        echo "No jupyter servers already running, try 'start' option."
        exit
    else
        ssh -i ${KEYFILE} -NfL 9999:localhost:9999 ${USER}@${HOST}
    fi
    url=$(echo $servers | grep -o 'http://localhost\S*')
    echo $url
    open $url

elif [ "$1" == 'stop' ]; then

    # get the new host address
    INSTANCE_STATUS=$(gcloud compute instances list --filter="name~""${INSTANCE_NAME}" | grep -e "${INSTANCE_NAME}" | grep -o '\d*\.\d*\.\d*\.\d*.*' | awk '{print $NF}')
    if [ ! $INSTANCE_STATUS == 'RUNNING' ]; then
      echo "Instance not running, Use command 'start' instead."
      exit
    fi
    HOST=$(gcloud compute instances list --filter="name~""${INSTANCE_NAME}" | grep -e "${INSTANCE_NAME}" | grep -o '\d*\.\d*\.\d*\.\d*' | tail -n1)

    echo "Stopping Jupyter notebook instances..."
    # get running instances and stop them
    ports=$(ssh -i "${KEYFILE}" -o "StrictHostKeyChecking no" ${USER}@${HOST} \
        'eval "$('\${HOME}/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"; conda activate '${ENV_NAME}'; jupyter notebook list' | ggrep -oP '(?<=:)[0-9]+' | uniq)
    echo $ports
    for port in $ports; do
        ssh -i "${KEYFILE}" -o "StrictHostKeyChecking no" ${USER}@${HOST} \
            'eval "$('\${HOME}/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"; conda activate '${ENV_NAME}'; jupyter notebook stop '${port}
    done
   
    # kill open ports left by ssh tunnels
    PIDS=$(lsof -i -P -n | grep 'ssh.*LISTEN' | awk '{print $2}' | uniq)    
    for pid in $PIDS; do
        kill $pid
    done
 
    echo "Stopping compute instance..."
    echo "Running command \"gcloud compute instances stop "${INSTANCE_NAME}"\""
    gcloud compute instances stop ${INSTANCE_NAME} > /dev/null
    
    echo "Shutdown complete."

elif [ "$1" == 'ssh-connect' ]; then
    
    # get the new host address
    INSTANCE_STATUS=$(gcloud compute instances list --filter="name~""${INSTANCE_NAME}" | grep -e "${INSTANCE_NAME}" | grep -o '\d*\.\d*\.\d*\.\d*.*' | awk '{print $NF}')
    if [ ! $INSTANCE_STATUS == 'RUNNING' ]; then
      echo "Instance not running, Use command 'start' instead."
      exit
    fi
    HOST=$(gcloud compute instances list --filter="name~""${INSTANCE_NAME}" | grep -e "${INSTANCE_NAME}" | grep -o '\d*\.\d*\.\d*\.\d*' | tail -n1)

    # ssh
    echo "Connecting to compute instance..."
    ssh -i ${KEYFILE} -A ${USER}@${HOST} 

elif [ "$1" == 'port-forward' ]; then

    # get the new host address
    INSTANCE_STATUS=$(gcloud compute instances list --filter="name~""${INSTANCE_NAME}" | grep -e "${INSTANCE_NAME}" | grep -o '\d*\.\d*\.\d*\.\d*.*' | awk '{print $NF}')
    if [ ! $INSTANCE_STATUS == 'RUNNING' ]; then
      echo "Instance not running, Use command 'start' instead."
      exit
    fi
    HOST=$(gcloud compute instances list --filter="name~""${INSTANCE_NAME}" | grep -e "${INSTANCE_NAME}" | grep -o '\d*\.\d*\.\d*\.\d*' | tail -n1)

    # ssh with port forwarding
    PORT=$ENV_NAME
    echo "Connecting to compute instance..."
    ssh -i ${KEYFILE} -A -NfL ${PORT}:localhost:${PORT} -o "StrictHostKeyChecking no" ${USER}@${HOST}

else
    
    echo "<cmd> must be 'start', 'reconnect', 'stop', 'ssh-connect' or 'port-forward'"

fi