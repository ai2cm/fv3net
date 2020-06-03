# copied from fv3kube micro-package to avoid package dependencies on fv3config and vcm


def list_jobs(client, job_labels):
    selector_format_labels = [f"{key}={value}" for key, value in job_labels.items()]
    combined_selectors = ",".join(selector_format_labels)
    jobs = client.list_job_for_all_namespaces(label_selector=combined_selectors)
    return jobs.items


def job_failed(job):
    conds = job.status.conditions
    conditions = [] if conds is None else conds
    for cond in conditions:
        if cond.status == "True":
            return cond.type == "Failed"
    return False


def job_complete(job):
    conds = job.status.conditions
    conditions = [] if conds is None else conds
    for cond in conditions:
        if cond.status == "True":
            return cond.type == "Complete"
    return False
