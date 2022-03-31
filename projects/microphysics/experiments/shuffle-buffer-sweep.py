import sys

sys.path.insert(0, "../argo")

from end_to_end import TrainingJob, load_yaml, submit_jobs  # noqa: E402


def _get_job(shuffle_buffer_size):
    config = load_yaml(f"../train/rnn.yaml")
    # Training loss problems are visible after 5 epochs
    config["epochs"] = 5
    config["shuffle_buffer_size"] = shuffle_buffer_size
    return TrainingJob(
        name=f"training-buffer-{shuffle_buffer_size}",
        fv3fit_image_tag=revision,
        config=config,
    )


revision = sys.argv[1] if len(sys.argv) > 1 else "latest"
jobs = [_get_job(n) for n in [10_000, 100_000, 1_000_000]]
submit_jobs(jobs, f"shuffle-buffer-experiments")
