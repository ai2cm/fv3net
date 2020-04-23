import kfp
from kfp import dsl

TAG = "738f327e3e5cdf31dc1eae630cc1f790873a58f9"

@dsl.pipeline(name='my-pipeline')
def pipeline(grid_spec: str, rundir: str, output: str):

    compute_diags_op = dsl.ContainerOp(
        name="compute diagnostics",
        image=f"us.gcr.io/vcm-ml/fv3net:{TAG}",
        command=["python", "workflows/prognostic_run_diags/save_prognostic_run_diags.py"],
        arguments=[
            "--grid-spec", grid_spec,
            rundir, output
        ]
    )

    metrics_op = dsl.ContainerOp(
        name="Prognostic run metrics",
        image="us.gcr/vcm-ml/metrics:v0.1.0",
        command=["bash", "-c"],
        arguments=[
            """
            python3 /metrics.py %s > /metrics.json
            """%(output)
        ],
        file_outputs={
            'output': '/metrics.json'
        }
    )

    metrics_op.after(compute_diags_op)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, __file__ + '.yaml')
