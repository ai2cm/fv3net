import kfp
from kfp import dsl

TAG = "738f327e3e5cdf31dc1eae630cc1f790873a58f9"


copy_from_gcs = kfp.components.load_component_from_url(
    "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/google-cloud/storage/download_dir/component.yaml"
)
GRID = "gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy/grid_spec"


@dsl.pipeline(name="my-pipeline")
def pipeline(rundir: str, output: str, grid: str = GRID):

    dl_atmos_op = copy_from_gcs("%s/atmos_dt_atmos.tile?.nc" % rundir)
    dl_grid_spec = copy_from_gcs("%s.tile?.nc" % grid)

    compute_diags_op = dsl.ContainerOp(
        name="compute diagnostics",
        image=f"us.gcr.io/vcm-ml/fv3net:{TAG}",
        command=[
            "python",
            "workflows/prognostic_run_diags/save_prognostic_run_diags.py",
        ],
        arguments=[
            "--grid-spec",
            "%s/grid_spec" % dl_grid_spec.output,
            dl_atmos_op.output,
            "/diags.nc",
        ],
        file_outputs={"diags": "/diags.nc"},
    )

    metrics_op = dsl.ContainerOp(
        name="Prognostic run metrics",
        image="us.gcr/vcm-ml/metrics:v0.1.0",
        command=["bash", "-c"],
        arguments=[
            """
            python3 /metrics.py %s > /metrics.json
            """
            % (compute_diags_op.output)
        ],
        file_outputs={"output": "/metrics.json"},
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pipeline, __file__ + ".yaml")
