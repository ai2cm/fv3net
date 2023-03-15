import sys
import subprocess

if __name__ == "__main__":
    job_name = sys.argv[1]
    p1 = subprocess.Popen(["argo", "get", job_name,], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(
        ["grep", job_name + "-"], stdin=p1.stdout, stdout=subprocess.PIPE
    )

    pod_name = subprocess.check_output(["awk", "{print $4}"], stdin=p2.stdout).strip()
    print("pod name: {}".format(pod_name.decode("utf-8")))

    p3 = subprocess.Popen(["podlogs", pod_name], stdout=subprocess.PIPE)

    p4 = subprocess.Popen(
        ["grep", "View run at"], stdin=p3.stdout, stdout=subprocess.PIPE
    )

    wandb_url = subprocess.check_output(["awk", "{print $5}"], stdin=p4.stdout).strip()
    print("wandb url: {}".format(wandb_url.decode("utf-8")))
