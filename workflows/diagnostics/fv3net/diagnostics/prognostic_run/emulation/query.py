import wandb
from dataclasses import dataclass


@dataclass
class PrognosticRunClient:
    tag: str
    project: str
    entity: str
    api: wandb.Api

    def use_artifact_in(self, run):
        """Gets prognostic rundir artifact.

        Makes an explicit link the wandb web console.
        """
        artifact_name = self.tag + ":latest"
        return run.use_artifact(artifact_name)

    def _get_run(self):
        """Return wandb.Run object for this prognostic run"""
        runs = self.api.runs(
            filters={"group": self.tag}, path=f"{self.entity}/{self.project}"
        )
        prognostic_runs = []

        for run in runs:
            if run.job_type == "prognostic_run":
                prognostic_runs.append(run)
        (run,) = prognostic_runs
        return run

    def get_rundir_url(self) -> str:
        run = self._get_run()
        return run.config["rundir"]
